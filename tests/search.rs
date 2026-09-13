//! Search-level correctness. Perft proves move generation; these prove the
//! search does not corrupt state, hallucinate mates, or play illegal moves.

use chess::board::Board;
use chess::chess_move::MoveList;
use chess::eval::{self, PstEval};
use chess::movegen::{generate, GenType};
use chess::search::{Limits, Params, Searcher, Shared, ThreadData};

fn search(fen: &str, depth: u32) -> (String, i32, u64) {
    chess::init();
    let b = Board::from_fen(fen).unwrap();
    let shared = Shared::new(16);
    let mut s = Searcher::new(&shared, ThreadData::new(0), PstEval, Params::default());
    let r = s.go(&b, &[], &Limits { depth: Some(depth), ..Default::default() }, None);
    (r.best_move.to_uci(), r.score, r.nodes)
}

#[test]
fn finds_forced_mates() {
    // Mate distance is the real property under test. The first move is only
    // asserted where the mate has a unique solution — the ladder mate below has
    // four (Ra7, Rb7, Ra8+, Rb8+ all mate in 2), so pinning it to one move would
    // test move ordering rather than correctness.
    let cases: &[(&str, u32, i32, &[&str])] = &[
        // Back-rank mate in 1: the black king is boxed in by its own pawns.
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3, 1, &["a1a8"]),
        // Two-rook ladder against a bare king, mate in 2.
        ("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1", 5, 2, &[]),
        // Mate in 1 from the other side, to check the sign of mate scores.
        ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", 3, 1, &["a8a1"]),
    ];
    for (fen, depth, mate_in, ok_moves) in cases {
        let (mv, score, _) = search(fen, *depth);
        assert!(
            eval::is_mate_score(score) && score > 0,
            "{fen}: expected a winning mate score, got {score} (move {mv})"
        );
        let plies = eval::MATE - score;
        assert_eq!(
            (plies + 1) / 2,
            *mate_in,
            "{fen}: expected mate in {mate_in}, got mate in {}",
            (plies + 1) / 2
        );
        if !ok_moves.is_empty() {
            assert!(ok_moves.contains(&mv.as_str()), "{fen}: unexpected move {mv}");
        }
    }
}

#[test]
fn recognises_being_mated_and_stalemated() {
    chess::init();
    // Already checkmated: no legal moves, in check.
    let b = Board::from_fen("7k/5QQ1/8/8/8/8/8/K7 b - - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert!(l.is_empty() && b.in_check(), "should be checkmate");

    // Stalemate: no legal moves, not in check.
    let b = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut l = MoveList::new();
    generate(&b, GenType::All, &mut l);
    assert!(l.is_empty() && !b.in_check(), "should be stalemate");
}

/// The engine must never take a defended pawn with its queen. The knight on c6
/// covers d4, so Qxd4?? loses the queen to Nxd4 — a refutation that only exists
/// one ply beyond the capture, which is exactly what quiescence is for.
#[test]
fn quiescence_refutes_hanging_capture() {
    let (mv, score, _) = search("4k3/8/2n5/8/3p4/8/8/3QK3 w - - 0 1", 6);
    assert_ne!(mv, "d1d4", "took a poisoned pawn that loses the queen");
    assert!(score > -200, "should not evaluate this position as lost: {score}");
}

/// A full game against itself. Catches illegal-move generation, state
/// corruption in copy-make, and any panic reachable from real play — none of
/// which perft exercises, since perft never runs the search.
#[test]
fn self_play_game_is_legal_throughout() {
    chess::init();
    let shared = Shared::new(16);
    let mut b = Board::startpos();
    let mut history: Vec<u64> = Vec::new();
    let limits = Limits { depth: Some(5), ..Default::default() };

    for ply in 0..160 {
        let mut legal = MoveList::new();
        generate(&b, GenType::All, &mut legal);
        if legal.is_empty() || b.halfmove() >= 100 {
            break;
        }
        let mut s = Searcher::new(&shared, ThreadData::new(0), PstEval, Params::default());
        let r = s.go(&b, &history, &limits, None);

        assert!(
            legal.contains(r.best_move),
            "ply {ply}: engine returned illegal move {} in {}",
            r.best_move,
            b.to_fen()
        );
        assert!(
            r.score.abs() <= eval::MATE,
            "ply {ply}: score {} out of range",
            r.score
        );

        history.push(b.key());
        b = b.make_move(r.best_move);
        assert_eq!(
            b.key(),
            b.recompute_key(),
            "ply {ply}: zobrist drifted after {}",
            r.best_move
        );
        // FEN must survive a round trip, which checks every board field.
        let fen = b.to_fen();
        assert_eq!(Board::from_fen(&fen).unwrap().to_fen(), fen);
    }
}

/// Searching the same position twice must give the same answer. A mismatch
/// means the TT is returning scores that depend on search path (usually a
/// mate-score adjustment bug or a bound used at the wrong depth).
#[test]
fn search_is_deterministic() {
    let a = search("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 7);
    let b = search("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 7);
    assert_eq!(a, b, "same position searched twice gave different results");
}

/// Deeper search should not lose material outright in a simple tactical spot,
/// and increasing depth must not crash or regress into an illegal move.
#[test]
fn depth_scaling_is_stable() {
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4";
    chess::init();
    let b = Board::from_fen(fen).unwrap();
    let mut legal = MoveList::new();
    generate(&b, GenType::All, &mut legal);
    for d in 1..=8 {
        let (mv, _, nodes) = search(fen, d);
        let m = chess::uci::parse_move(&b, &mv).expect("move must parse");
        assert!(legal.contains(m), "depth {d}: illegal move {mv}");
        assert!(nodes > 0);
    }
}

/// A FEN claiming a castling right the position cannot back must not be able
/// to crash the engine.
///
/// `from_fen` used to take "KQkq" at its word. Movegen then generated the
/// castle, `make_move` read the rook off an empty corner, and
/// `Piece::NONE.piece_type()` panicked — with `panic = "abort"` in the release
/// profile that killed the whole process, taking the GUI server down with it.
/// Hand-editing a FEN to remove a rook, which is the ordinary way to set up a
/// material-odds game, was enough to trigger it.
#[test]
fn castling_rights_are_sanitised_on_input() {
    chess::init();
    // Black's rooks are gone but the FEN still claims kingside and queenside.
    let fen = "1nbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
    let b = Board::from_fen(fen).unwrap();
    assert_eq!(b.castling_rights() & 0b1100, 0, "black kept a right it cannot back");
    assert_ne!(b.castling_rights() & 0b0011, 0, "white's rights are real and must survive");
    // The searches below are the actual regression: both used to abort.
    let (mv, _, _) = search(fen, 6);
    assert!(!mv.is_empty() && mv != "0000", "no move from a sanitised position");

    // A king off its home square cannot castle either, however the FEN reads.
    let moved_king = "rnbq1bnr/ppppkppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
    assert_eq!(Board::from_fen(moved_king).unwrap().castling_rights() & 0b1100, 0);

    // And a legitimate position must keep every right it is entitled to.
    let full = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    assert_eq!(Board::from_fen(full).unwrap().castling_rights() & 0b1111, 0b1111);
}

/// The work ceiling binds, and binds reproducibly.
///
/// The whole point of a work budget is that it is a budget: two runs of the
/// same position at the same ceiling must spend the same amount and reach the
/// same move, or an objective scored against it is measuring the scheduler.
/// `--features work` only, because a playing build has no counter to test.
///
/// Kiwipete rather than the start position: from the opening a single
/// repetition counts as a draw (see `is_draw`), the tree collapses, and the
/// search reaches MAX_PLY in under 5000 nodes without ever touching a budget
/// this size. A ceiling that is never reached tests nothing.
#[cfg(feature = "work")]
#[test]
fn work_limit_binds_and_is_reproducible() {
    use chess::board::Board;
    use chess::qeval::DefaultEval;
    use chess::search::{Limits, Params, Searcher, Shared, ThreadData};

    const KIWIPETE: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let b = Board::from_fen(KIWIPETE).unwrap();

    // Deliberately does NOT install a price table: a fresh `ThreadData` must
    // already be priced. The first version of this test installed one, so it
    // passed while `tune` -- which builds its own `ThreadData` -- ran to
    // `MAX_PLY` because the ceiling never bound. A test that arranges the
    // condition the caller cannot arrange proves nothing.
    let run = |work: Option<u64>| {
        let limits = Limits { work, depth: Some(20), ..Default::default() };
        let shared = Shared::new(16);
        let td = ThreadData::new(0);
        let mut s = Searcher::new(&shared, td, DefaultEval::default(), Params::default());
        let r = s.go(&b, &[], &limits, None);
        (r.best_move, r.nodes, s.td.prof.work_ps)
    };

    // Calibrate against the position rather than hard-coding a budget: search
    // it free first, then ask for a quarter of what that cost. A fixed
    // constant would silently stop binding the moment the eval or the price
    // list changed, and a ceiling that is never reached tests nothing. Both
    // earlier versions of this test picked a constant and failed for that
    // reason -- loudly, which is the only reason the budget is trusted here.
    let free = run(None);
    let budget = chess::work::units(free.2 as f64 / 1000.0) as u64 / 4;
    assert!(budget > 100, "free run only cost {budget} units; nothing to bound");

    let a = run(Some(budget));
    let b2 = run(Some(budget));
    let spent = chess::work::units(a.2 as f64 / 1000.0);

    // It stopped, and it stopped ON the ceiling: the test is per node, so the
    // overshoot is bounded by what one node can cost.
    assert!(
        spent >= budget as f64 && spent < budget as f64 * 1.02,
        "spent {spent} units against a {budget} budget"
    );
    // The ceiling is doing the stopping, not the depth limit behind it.
    assert!(
        free.2 > a.2 * 2,
        "unbudgeted run spent {} ps, budgeted {} -- the ceiling never bound",
        free.2,
        a.2
    );
    // Reproducible to the last picosecond, which is the property the whole
    // method rests on.
    assert_eq!(a, b2, "work-limited search is not reproducible");
}
