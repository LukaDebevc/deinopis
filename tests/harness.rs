//! Tests for the measurement harness: notation, opening book, game-end rules
//! and match statistics.
//!
//! These matter for the same reason perft matters. A typo in the book, an
//! off-by-one in repetition detection or a wrong variance in the SPRT does not
//! crash — it quietly produces a *number*, and the number is what every later
//! decision in the project rests on.

use chess::board::Board;
use chess::chess_move::MoveList;
use chess::game::{self, Status};
use chess::movegen::{generate, GenType};
use chess::san::san;
use chess::sprt::*;

fn setup() {
    chess::init();
}

/// Play a UCI line from the start position, returning the final board and the
/// keys of every position before it.
fn play(line: &str) -> (Board, Vec<u64>) {
    let mut b = Board::startpos();
    let mut keys = Vec::new();
    for tok in line.split_whitespace() {
        let mut list = MoveList::new();
        generate(&b, GenType::All, &mut list);
        let m = list
            .find_uci(tok)
            .unwrap_or_else(|| panic!("illegal move '{tok}' in line '{line}'"));
        keys.push(b.key());
        b = b.make_move(m);
    }
    (b, keys)
}

// ------------------------------------------------------------------ book

#[test]
fn every_book_line_is_legal() {
    setup();
    for line in chess::book::LINES {
        let mut b = Board::startpos();
        for (i, tok) in line.split_whitespace().enumerate() {
            let mut list = MoveList::new();
            generate(&b, GenType::All, &mut list);
            match list.find_uci(tok) {
                Some(m) => b = b.make_move(m),
                None => panic!("book line '{line}': move {} ('{tok}') is illegal", i + 1),
            }
        }
        // A book line that ends the game is not an opening.
        assert_eq!(
            game::status(&b, &[]),
            Status::Ongoing,
            "book line '{line}' ends the game"
        );
    }
}

// The default book is compiled in, so a bad line in `books/lich.epd` is a
// build-time asset that only fails when a match starts 40 minutes deep. Parse
// it here instead, and check the two properties a book must have: every start
// position is legal with a move available, and no two lines are the same
// position -- the second is what makes the printed SE honest (LEDGER 112).
#[test]
fn default_book_is_legal_and_distinct() {
    setup();
    let book = chess::book::default_book();
    assert!(book.len() >= 1500, "default book has {} lines; below ~1500 the design effect returns", book.len());
    let mut fens: Vec<String> = Vec::with_capacity(book.len());
    for o in &book {
        let b = Board::from_fen(&o.start_fen)
            .unwrap_or_else(|e| panic!("default book: bad FEN '{}': {e}", o.start_fen));
        assert!(o.moves.is_empty(), "default book is EPD; line carries moves");
        assert_eq!(
            game::status(&b, &[]),
            Status::Ongoing,
            "default book position '{}' is already over",
            o.start_fen
        );
        fens.push(b.to_fen());
    }
    let n = fens.len();
    fens.sort();
    fens.dedup();
    assert_eq!(fens.len(), n, "default book repeats {} positions", n - fens.len());
}

#[test]
fn book_lines_are_distinct() {
    setup();
    let mut fens: Vec<String> = chess::book::builtin()
        .iter()
        .map(|l| play(&l.moves.join(" ")).0.to_fen())
        .collect();
    let n = fens.len();
    fens.sort();
    fens.dedup();
    assert_eq!(fens.len(), n, "book contains duplicate final positions");
}

// ------------------------------------------------------------------ san

#[test]
fn san_basic_moves() {
    setup();
    let b = Board::startpos();
    let named = |uci: &str| {
        let mut l = MoveList::new();
        generate(&b, GenType::All, &mut l);
        san(&b, l.find_uci(uci).unwrap())
    };
    assert_eq!(named("e2e4"), "e4");
    assert_eq!(named("g1f3"), "Nf3");
    assert_eq!(named("b1c3"), "Nc3");
}

#[test]
fn san_disambiguates() {
    setup();
    // Two rooks on the first rank both reach d1: file disambiguation.
    let b = Board::from_fen("4k3/8/4K3/8/8/8/8/R6R w - - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("a1d1").unwrap()), "Rad1");
    assert_eq!(san(&b, l.find_uci("h1d1").unwrap()), "Rhd1");

    // Two rooks on the same file: rank disambiguation.
    let b = Board::from_fen("R7/8/8/4k3/8/8/8/R6K w - - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("a1a4").unwrap()), "R1a4");
    assert_eq!(san(&b, l.find_uci("a8a4").unwrap()), "R8a4");
}

#[test]
fn san_castling_promotion_ep_and_mate() {
    setup();
    let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("e1g1").unwrap()), "O-O");
    assert_eq!(san(&b, l.find_uci("e1c1").unwrap()), "O-O-O");

    let b = Board::from_fen("1n6/P7/8/8/8/8/8/2k1K3 w - - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("a7a8q").unwrap()), "a8=Q");
    assert_eq!(san(&b, l.find_uci("a7b8n").unwrap()), "axb8=N");

    // En passant is a pawn capture and is written like one.
    let (b, _) = play("e2e4 a7a6 e4e5 d7d5");
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("e5d6").unwrap()), "exd6");

    // Fool's mate gets the '#'.
    let (b, _) = play("f2f3 e7e5 g2g4");
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert_eq!(san(&b, l.find_uci("d8h4").unwrap()), "Qh4#");
}

#[test]
fn san_line_reads_as_a_game() {
    setup();
    let b = Board::startpos();
    let mut moves = Vec::new();
    let mut cur = b;
    for tok in "e2e4 e7e5 g1f3 b8c6 f1b5".split_whitespace() {
        let mut l = MoveList::new();
        generate(&cur, GenType::All, &mut l);
        let m = l.find_uci(tok).unwrap();
        moves.push(m);
        cur = cur.make_move(m);
    }
    assert_eq!(
        chess::san::movetext(&b, &moves, 0),
        "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    );
}

// ------------------------------------------------------------------ game rules

#[test]
fn threefold_repetition_needs_three() {
    setup();
    // Two knight round-trips return the start position for the third time.
    let (b, keys) = play("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
    assert_eq!(game::status(&b, &keys), Status::Repetition);

    // One round-trip is only the second occurrence: not a draw yet.
    let (b, keys) = play("g1f3 g8f6 f3g1 f6g8");
    assert_eq!(game::status(&b, &keys), Status::Ongoing);
}

#[test]
fn fifty_move_and_material() {
    setup();
    let b = Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 100 80").unwrap();
    assert_eq!(game::status(&b, &[]), Status::FiftyMove);

    for fen in [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KB2 w - - 0 1",
        "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
    ] {
        let b = Board::from_fen(fen).unwrap();
        assert_eq!(
            game::status(&b, &[]),
            Status::InsufficientMaterial,
            "{fen}"
        );
    }
    // Two bishops can mate, so the game is not dead.
    let b = Board::from_fen("4k3/8/8/8/8/8/8/3BKB2 w - - 0 1").unwrap();
    assert_eq!(game::status(&b, &[]), Status::Ongoing);
}

#[test]
fn checkmate_and_stalemate_are_distinguished() {
    setup();
    let (b, _) = play("f2f3 e7e5 g2g4 d8h4");
    assert_eq!(game::status(&b, &[]), Status::Checkmate);
    let b = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    assert_eq!(game::status(&b, &[]), Status::Stalemate);
}

// ------------------------------------------------------------------ statistics

#[test]
fn elo_score_roundtrip() {
    assert!((elo_to_score(0.0) - 0.5).abs() < 1e-12);
    for elo in [-400.0, -100.0, -5.0, 0.0, 5.0, 100.0, 400.0] {
        assert!((score_to_elo(elo_to_score(elo)) - elo).abs() < 1e-9, "{elo}");
    }
    // The definitional anchor: +400 Elo is a 10:1 expected score.
    assert!((elo_to_score(400.0) - 10.0 / 11.0).abs() < 1e-12);
}

#[test]
fn phi_is_a_cdf() {
    assert!((phi(0.0) - 0.5).abs() < 1e-6);
    assert!((phi(1.96) - 0.975).abs() < 1e-3);
    assert!((phi(-1.96) - 0.025).abs() < 1e-3);
    assert!(phi(-6.0) < 1e-6 && phi(6.0) > 1.0 - 1e-6);
}

#[test]
fn even_match_is_inconclusive_and_lopsided_one_is_not() {
    let sprt = Sprt::new(0.0, 5.0);

    // A dead-even match must not accept H1 no matter how long it runs.
    let mut even = MatchStats::default();
    for i in 0..500 {
        even.add_pair(if i % 2 == 0 {
            PairResult::WinDraw
        } else {
            PairResult::LossDraw
        });
        even.add_game(0.5);
        even.add_game(0.5);
    }
    assert_ne!(sprt.verdict(&even), SprtVerdict::AcceptH1);
    let (elo, _, _) = even.elo().unwrap();
    assert!(elo.abs() < 5.0, "even match reported {elo} Elo");

    // A one-sided match must reach H1, and quickly.
    let mut strong = MatchStats::default();
    let mut decided_after = None;
    for i in 0..500 {
        strong.add_pair(PairResult::WinDraw);
        strong.add_game(1.0);
        strong.add_game(0.5);
        if decided_after.is_none() && sprt.verdict(&strong) == SprtVerdict::AcceptH1 {
            decided_after = Some(i + 1);
        }
    }
    assert_eq!(sprt.verdict(&strong), SprtVerdict::AcceptH1);
    assert!(strong.los().unwrap() > 0.99);
    assert!(strong.elo().unwrap().0 > 100.0);

    // ...and a losing one must reach H0.
    let mut weak = MatchStats::default();
    for _ in 0..200 {
        weak.add_pair(PairResult::LossDraw);
        weak.add_game(0.0);
        weak.add_game(0.5);
    }
    assert_eq!(sprt.verdict(&weak), SprtVerdict::AcceptH0);
}

#[test]
fn llr_moves_in_the_right_direction() {
    let sprt = Sprt::new(0.0, 5.0);
    let mut winning = MatchStats::default();
    let mut losing = MatchStats::default();
    for _ in 0..50 {
        winning.add_pair(PairResult::WinDraw);
        winning.add_pair(PairResult::Balanced);
        losing.add_pair(PairResult::LossDraw);
        losing.add_pair(PairResult::Balanced);
    }
    assert!(sprt.llr(&winning).unwrap() > 0.0);
    assert!(sprt.llr(&losing).unwrap() < 0.0);
    // Bounds are symmetric for alpha == beta.
    let (lo, hi) = sprt.bounds();
    assert!((lo + hi).abs() < 1e-9);
}

#[test]
fn error_bars_shrink_with_games() {
    let mut short = MatchStats::default();
    let mut long = MatchStats::default();
    for i in 0..1000 {
        let p = if i % 3 == 0 {
            PairResult::WinDraw
        } else {
            PairResult::Balanced
        };
        if i < 25 {
            short.add_pair(p);
        }
        long.add_pair(p);
    }
    let (_, slo, shi) = short.elo().unwrap();
    let (_, llo, lhi) = long.elo().unwrap();
    assert!(
        (lhi - llo) < (shi - slo) / 4.0,
        "interval did not shrink: short {:.0}, long {:.0}",
        shi - slo,
        lhi - llo
    );
}

/// A promotion written with an uppercase piece letter must parse.
///
/// UCI specifies lowercase, but Simbelmyne 1.10.0 emits "a7a8Q" and a strict
/// parser scores that as an illegal move. In one 200-game gauntlet that was
/// 28 forfeits, all in our favour, which silently inflated the anchor by
/// hundreds of Elo. Leniency here is not sloppiness -- it is the difference
/// between measuring an opponent and measuring our own parser.
#[test]
fn promotion_uci_is_case_insensitive() {
    use chess::board::Board;
    use chess::chess_move::MoveList;
    use chess::movegen::{generate, GenType};

    let b = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let mut list = MoveList::new();
    generate(&b, GenType::All, &mut list);

    let lower = list.find_uci("a7a8q").expect("lowercase promotion must parse");
    let upper = list.find_uci("a7a8Q").expect("uppercase promotion must parse");
    assert_eq!(lower, upper);
    assert!(lower.is_promotion());

    // Case-insensitivity must not invent legal moves.
    assert!(list.find_uci("a7a8K").is_none());
    assert!(list.find_uci("h2h4").is_none());
}

// ------------------------------------------------------- book: the two formats

#[test]
fn book_reads_uci_move_lines() {
    setup();
    let op = chess::book::parse_line("e2e4 e7e5 g1f3").unwrap();
    assert!(op.from_startpos());
    assert_eq!(op.moves, vec!["e2e4", "e7e5", "g1f3"]);
}

#[test]
fn book_reads_epd_and_strips_opcodes() {
    setup();
    // Stockfish's books end the record with `;` opcodes; lichess's with `[0.0]`.
    // Both must yield the same position, and neither suffix may leak into the
    // FEN -- a stray token there becomes a bogus halfmove counter.
    let sf = chess::book::parse_line(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4 ; id \"x\";",
    )
    .unwrap();
    let li = chess::book::parse_line(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4 [0.0]",
    )
    .unwrap();
    assert_eq!(sf, li);
    assert!(!sf.from_startpos());
    assert!(sf.moves.is_empty());
    assert_eq!(chess::board::Board::from_fen(&sf.start_fen).unwrap().fullmove(), 4);
}

#[test]
fn book_epd_without_counters_gets_defaults() {
    setup();
    // A bare EPD has only four fields. It must still load, with the counters
    // defaulted rather than the line rejected.
    let op = chess::book::parse_line(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
    )
    .unwrap();
    let b = chess::board::Board::from_fen(&op.start_fen).unwrap();
    assert_eq!((b.halfmove(), b.fullmove()), (0, 1));
}

#[test]
fn book_rejects_a_position_with_no_move() {
    setup();
    // Stalemate: legal as a position, useless as an opening, and it would
    // otherwise be discovered as a 0-ply game in the middle of a match.
    let e = chess::book::parse_line("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap_err();
    assert!(e.contains("no legal move"), "unexpected error: {e}");
}

#[test]
fn book_rejects_a_bad_fen() {
    setup();
    assert!(chess::book::parse_line("not/a/fen w KQkq - 0 1").is_err());
}
