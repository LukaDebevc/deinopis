//! The UCI front-end, driven the way a GUI or a shell pipe drives it: the real
//! binary as a subprocess, commands on stdin, answers read back from stdout.
//! No net is passed, so these run on the PeSTO fallback; they test the
//! protocol, not the play.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run the engine, feed it `steps` (text to write, then a pause), close stdin,
/// and return everything it printed. Panics if it has not exited `limit`
/// after stdin closed, which is how a hang shows up as a failure rather than
/// a stuck test run.
fn run(args: &[&str], steps: &[(&str, u64)], limit: Duration) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_chess"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn engine");
    let mut out = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    {
        let mut stdin = child.stdin.take().unwrap();
        for (text, pause_ms) in steps {
            stdin.write_all(text.as_bytes()).unwrap();
            stdin.flush().unwrap();
            std::thread::sleep(Duration::from_millis(*pause_ms));
        }
    }
    let t = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if t.elapsed() > limit {
            let _ = child.kill();
            panic!("engine still running {limit:?} after end of input");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    reader.join().unwrap()
}

#[test]
fn uci_handshake_names_the_engine() {
    let out = run(&[], &[("uci\nquit\n", 0)], Duration::from_secs(5));
    let want = format!("id name Deinopis {}", env!("CARGO_PKG_VERSION"));
    assert!(out.contains(&want), "no `{want}`:\n{out}");
    assert!(out.contains("id author Luka Debevc"), "{out}");
    assert!(out.trim_end().ends_with("uciok"), "{out}");
}

#[test]
fn help_flag_prints_help() {
    for flag in ["--help", "-h"] {
        let out = run(&[flag], &[], Duration::from_secs(5));
        assert!(out.contains("USAGE"), "`{flag}` printed no usage:\n{out}");
    }
}

#[test]
fn end_of_input_still_prints_the_bestmove() {
    // A limited search runs to its limit; an open-ended one is stopped.
    for go in ["go depth 6", "go infinite", "go"] {
        let out = run(&[], &[(&format!("position startpos\n{go}\n"), 0)], Duration::from_secs(20));
        assert!(out.contains("bestmove"), "`{go}` then EOF printed no bestmove:\n{out}");
    }
}

#[test]
fn setoption_hash_during_a_search_does_not_hang() {
    let out = run(
        &[],
        &[
            ("uci\nposition startpos\ngo infinite\n", 300),
            ("setoption name Hash value 32\nstop\nisready\nquit\n", 0),
        ],
        Duration::from_secs(10),
    );
    assert!(out.contains("bestmove"), "no bestmove:\n{out}");
    assert!(out.contains("readyok"), "no readyok:\n{out}");
}
