//! A chess engine.
//!
//! Module map (see ARCHITECTURE.md for the why):
//!   types, bitboard, attacks   — board geometry, no game state
//!   zobrist, board, chess_move — position representation, copy-make
//!   movegen                    — legal move generation
//!   eval                       — static evaluation (hand-crafted for now)
//!   tt, search                 — transposition table and alpha-beta search
//!   game                       — the rules of when a game is over
//!   uci, gui                   — protocol front-end and a local web board
//!   perft                      — move generation correctness harness
//!   trace                      — per-decision record of what the price list did
//!   nodeprof                   — where a node's time goes (--features nodeprof)
//!   work                       — deterministic cost: counts x a frozen price
//!                                table, in place of the wall clock
//!   san, book, sprt, matchplay — the measurement harness: notation, openings,
//!                                match statistics, engine-vs-engine play

pub mod attacks;
pub mod book;
pub mod bitboard;
pub mod board;
pub mod chess_move;
pub mod movegen;
pub mod eval;
#[cfg(feature = "evalstats")]
pub mod evalstats;
#[cfg(feature = "checkstats")]
pub mod checkstats;
pub mod deepeval;
pub mod wdleval;
#[cfg(feature = "hyst")]
pub mod hyst;
#[cfg(feature = "movedump")]
pub mod movedump;
pub mod nodeprof;
pub mod work;
pub mod qeval;
pub mod game;
pub mod gui;
pub mod matchplay;
pub mod perft;
pub mod san;
pub mod search;
pub mod sprt;
pub mod trace;
pub mod tt;
pub mod tune;
pub mod types;
pub mod uci;
pub mod zobrist;

/// Build all runtime lookup tables. Must be called once, before anything else.
/// Idempotent and thread-safe.
pub mod adaptive;

pub fn init() {
    attacks::init();
    zobrist::init();
    adaptive::init_from_env();
    search::init_pot();
    search::init_corr();
    qeval::init_dual();
    eval::init_from_args();
}
