//! Stockfish binpack -> flat 32-byte records for the eval trainer.
//!
//! Why a separate crate: the engine has no dependencies on purpose, and
//! decoding a binpack is a few thousand lines nobody should rewrite. This is
//! the "NNUE trainer is a separate concern" carve-out in CLAUDE.md.
//!
//! Output is `bulletformat::ChessBoard`, 32 bytes, `#[repr(C)]`:
//!   occ: u64 | pcs: [u8;16] | score: i16 | result: u8 | ksq: u8 | opp_ksq: u8 | extra: [u8;3]
//!
//! The record is *side-to-move relative*: when Black is to move the board is
//! byte-swapped (vertical mirror) and the colours are exchanged, so the side to
//! move is always "White at the bottom". Score and result are stm-relative too.
//! That is the canonicalisation the quadratic form wants, and it halves the
//! data the model has to see.
//!
//! Games are written contiguously and in file order. That is deliberate: a
//! contiguous tail of the output is a *game-level* validation split. Splitting
//! by position instead would leak, since every position in a game carries the
//! same result label.
//!
//! `extra[3]` is unused by bulletformat (`from_raw` zeroes it, nothing reads
//! it), so it carries the game structure the flat format otherwise destroys:
//!   extra[0] bit0  this position passes `keep` (the training filter)
//!   extra[0] bit1  this position is the first ply of a game
//!   extra[1..3]    u16 ply, little-endian
//! With `--mark`, filtered positions are *flagged and written* rather than
//! dropped, so a game stays a contiguous run of plies. Anything that needs
//! game history (latched features, per-game splits) requires that; a trainer
//! that only wants the filtered set masks on bit0 and is unaffected.
//!
//! `--aux <file>` writes a second, parallel file: one little-endian u16 per
//! record, in the same order, carrying the state the 32-byte format drops.
//! lc0's input planes need all of it, and the 32-byte record has 6 spare bits
//! where 13 are wanted -- so a side file rather than a wider record, which
//! leaves `chess893.data` and everything trained on it untouched.
//!
//!   bit 0     the side to move can castle kingside
//!   bit 1     ... queenside
//!   bit 2     the other side can castle kingside
//!   bit 3     ... queenside
//!   bit 4     the side to move is Black
//!   bits 5-11 the rule-50 counter, saturating at 127
//!   bit 12    an en-passant capture is available
//!
//! En passant is one bit and not a square because the classical 112-plane
//! format has no en-passant plane at all -- lc0 conveys it through the history
//! planes, which we do not have. The bit is kept so a later study can measure
//! how much that costs rather than have to re-extract to ask.
//!
//! `--fen-dump <file>` writes the FEN of every written record, one per line.
//! It exists so the packing above can be checked against an independent
//! decoder: build lc0's input planes from (record, aux) and from the FEN and
//! require them to be identical. Use it with a small `--max`; it is many times
//! the size of the data.
//!
//! Castling is stated from the side to move's point of view because that is
//! what lc0's encoder wants and what the 32-byte record already does with the
//! board itself. Nothing here is derivable from the record: a mirrored Black
//! position and a White one are byte-identical.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;

use bulletformat::ChessBoard;
use sfbinpack::{
    chess::{color::Color, coords::Square, piecetype::PieceType, r#move::MoveType},
    read_chunk_into, ChunkReader, TrainingDataEntry,
};

/// The filter the Stockfish training pipeline uses, and that `bullet`'s own
/// example reproduces. Each clause removes positions whose label is not a
/// statement about the *static* value of the position:
///   - opening plies are book/randomised, not play;
///   - in check, the static eval is meaningless — search resolves it;
///   - |score| > 10000 is a mate score, which no eval can or should fit;
///   - a capture or a non-quiet best move means the position is mid-exchange,
///     so the search score reflects the capture, not the position.
fn keep(e: &TrainingDataEntry) -> bool {
    e.ply >= 16
        && !e.pos.is_checked(e.pos.side_to_move())
        && e.score.unsigned_abs() <= 10_000
        && e.mv.mtype() == MoveType::Normal
        && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None
}

/// Verbatim from `bullet_lib`'s sfbinpack loader, so our records are
/// byte-compatible with bulletformat tooling.
///
/// Note the apparent double flip: `entry.score`/`entry.result` are already
/// stm-relative in the binpack, so they are converted to White-relative here
/// and `from_raw` flips them back. Net effect is stm-relative, which is what we
/// want; removing "just one" of the two flips silently mislabels every Black
/// position.
fn to_record(e: &TrainingDataEntry) -> ChessBoard {
    let mut bbs = [0u64; 8];
    let stm = usize::from(e.pos.side_to_move().ordinal());
    let pc_bb = |pt| e.pos.pieces_bb_color(Color::Black, pt).bits() | e.pos.pieces_bb_color(Color::White, pt).bits();

    bbs[0] = e.pos.pieces_bb(Color::White).bits();
    bbs[1] = e.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = e.score;
    let mut result = f32::from(1 + e.result) / 2.0;
    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).expect("malformed binpack entry")
}

/// The side-to-move-relative state the 32-byte record cannot hold. See the
/// module header for the bit layout.
fn to_aux(e: &TrainingDataEntry) -> u16 {
    use sfbinpack::chess::castling_rights::CastlingRights as CR;
    let black = e.pos.side_to_move() == Color::Black;
    let cr = e.pos.castling_rights();
    let (we_k, we_q, th_k, th_q) = if black {
        (CR::BLACK_KING_SIDE, CR::BLACK_QUEEN_SIDE, CR::WHITE_KING_SIDE, CR::WHITE_QUEEN_SIDE)
    } else {
        (CR::WHITE_KING_SIDE, CR::WHITE_QUEEN_SIDE, CR::BLACK_KING_SIDE, CR::BLACK_QUEEN_SIDE)
    };
    let ep = e.pos.ep_square() != Square::NONE;
    u16::from(cr.contains(we_k))
        | u16::from(cr.contains(we_q)) << 1
        | u16::from(cr.contains(th_k)) << 2
        | u16::from(cr.contains(th_q)) << 3
        | u16::from(black) << 4
        | (e.pos.rule50_counter().min(127)) << 5
        | u16::from(ep) << 12
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let input = arg("--input").expect("--input <file.binpack>");
    let output = arg("--output").expect("--output <file.data>");
    let max: u64 = arg("--max").map(|s| s.parse().expect("--max")).unwrap_or(u64::MAX);
    // Keep at most one position in `stride` from each game. Positions inside a
    // game are highly correlated, so a stride buys diversity per byte written.
    // This thins plies *within* games, which destroys game continuity.
    let stride: u64 = arg("--stride").map(|s| s.parse().expect("--stride")).unwrap_or(1);
    // Keep every `game_stride`-th game with all of its plies. This is the
    // stride to use when the study needs whole games rather than loose
    // positions: it buys diversity by dropping games, not by punching holes.
    let game_stride: u64 = arg("--game-stride").map(|s| s.parse().expect("--game-stride")).unwrap_or(1);
    // Write positions that fail `keep`, flagged in extra[0] bit0, instead of
    // dropping them. 46% of plies fail the filter, and dropping them leaves a
    // game unreconstructable.
    let mark = env::args().any(|a| a == "--mark");
    assert!(
        !(mark && stride != 1),
        "--mark writes every ply of a kept game; --stride punches holes in exactly that. \
         Use --game-stride to thin instead."
    );

    let mut reader = BufReader::with_capacity(1 << 22, File::open(&input).expect("open input"));
    // Records are fixed-size, so appending months into one file is just
    // concatenation. The validation tail stays a game-level split either way.
    let append = env::args().any(|a| a == "--append");
    let f = if append {
        OpenOptions::new().create(true).append(true).open(&output).expect("open output")
    } else {
        File::create(&output).expect("create output")
    };
    let mut out = BufWriter::with_capacity(1 << 22, f);
    let mut aux = arg("--aux").map(|path| {
        let f = if append {
            OpenOptions::new().create(true).append(true).open(&path).expect("open aux")
        } else {
            File::create(&path).expect("create aux")
        };
        BufWriter::with_capacity(1 << 21, f)
    });
    let mut fend = arg("--fen-dump").map(|path| {
        BufWriter::with_capacity(1 << 20, File::create(&path).expect("create fen dump"))
    });

    let mut chunk = Vec::new();
    let mut seen: u64 = 0;
    let mut kept: u64 = 0;
    let mut written: u64 = 0;
    let mut selected: u64 = 0;
    let mut games_seen: u64 = 0;
    let mut games_written: u64 = 0;
    // KNOWN, DELIBERATELY UNFIXED: `ply <= prev_ply` is false for the first
    // entry of the file (ply 1 vs -1), so `game_selected` stays false until the
    // SECOND game starts and the first game of a binpack is dropped. One game
    // in 801k. It is left alone because `chess893.data` and every net trained
    // on it were produced this way, and changing it would shift every record by
    // one game -- which would silently break the `--aux` side file's alignment
    // with data already on disk. Fix it only together with a full re-extract.
    let mut prev_ply: i64 = -1;
    let mut game_selected = false;
    let start = Instant::now();

    // A truncated trailing chunk is an error, not EOF. That is the normal shape
    // of a byte-range prefix of a binpack, so stop cleanly and say so rather
    // than panicking and losing the buffered tail of the output.
    'outer: loop {
        match read_chunk_into(&mut reader, &mut chunk) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                eprintln!("warning: stopping at unreadable chunk ({e}); input is truncated or corrupt");
                break;
            }
        }
        let mut cr = ChunkReader::default();
        while cr.has_next(&chunk) {
            let e = cr.next(&chunk);
            seen += 1;

            // Game boundary. Measured over 95.7M entries / 801k games of
            // fishpack32: every within-game ply delta is exactly +1 and every
            // game starts at ply 1, with zero exceptions. So this is an exact
            // boundary, not a guess from piece-count jumps.
            let ply = i64::from(e.ply);
            let is_game_start = ply <= prev_ply;
            prev_ply = ply;
            if is_game_start {
                games_seen += 1;
                game_selected = games_seen % game_stride == 0;
                if game_selected {
                    games_written += 1;
                }
            }
            if !game_selected {
                continue;
            }

            selected += 1;
            let passes = keep(&e);
            if passes {
                kept += 1;
            }
            if !mark {
                if !passes {
                    continue;
                }
                if kept % stride != 0 {
                    continue;
                }
            }

            let mut rec = to_record(&e);
            rec.extra = [
                u8::from(passes) | (u8::from(is_game_start) << 1),
                (e.ply & 0xFF) as u8,
                (e.ply >> 8) as u8,
            ];
            // SAFETY: ChessBoard is #[repr(C)], 32 bytes, no padding-dependent
            // invariants and no pointers. Reinterpreting it as bytes is the
            // documented on-disk representation of bulletformat.
            let bytes: &[u8; 32] = unsafe { &*(&rec as *const ChessBoard as *const [u8; 32]) };
            out.write_all(bytes).expect("write");
            if let Some(a) = aux.as_mut() {
                a.write_all(&to_aux(&e).to_le_bytes()).expect("write aux");
            }
            if let Some(d) = fend.as_mut() {
                writeln!(d, "{}", e.pos.fen().expect("fen")).expect("write fen");
            }
            written += 1;

            if written % 4_000_000 == 0 {
                let s = start.elapsed().as_secs_f64();
                eprintln!(
                    "{written:>12} written  {kept:>12} kept  {seen:>12} seen  {:.0} pos/s",
                    seen as f64 / s
                );
            }
            if written >= max {
                break 'outer;
            }
        }
    }

    out.flush().expect("flush");
    if let Some(a) = aux.as_mut() {
        a.flush().expect("flush aux");
    }
    if let Some(d) = fend.as_mut() {
        d.flush().expect("flush fen dump");
    }
    let s = start.elapsed().as_secs_f64();
    // The pass rate is over the entries actually considered, i.e. those in
    // selected games. Dividing by `seen` would fold in the game stride and
    // report a filter rate that is really a sampling rate.
    eprintln!(
        "done: {written} records ({} MB) written, {games_written}/{games_seen} games kept, \
         {kept}/{selected} ({:.1}%) of considered entries pass the filter, {seen} entries read, {:.0}s",
        written * 32 / 1_000_000,
        100.0 * kept as f64 / selected.max(1) as f64,
        s
    );
}
