//! Diagnostic: bow a string at constant speed/pressure and print the
//! amplitude envelope over time — does it reach a steady state, and
//! where does that state sit relative to the safety clamp?
//!
//!     cargo run --release --example bowprobe

use timber::stream::{Bank, Params};
use timber::util::{SR, measured_freq};

fn main() {
    let grid = [0.1, 0.3, 0.5, 0.7, 0.9];
    for (speed, pressure) in grid.iter().flat_map(|&s| grid.iter().map(move |&p| (s, p))) {
        let mut bank = Bank::new(&[220.0]);
        let p = Params {
            decay: 0.996,
            damping: 0.5,
            stiffness: 0.0,
            level: 1.0,
            bow_on: true,
            couple: 0.0,
            bow_speed: speed,
            bow_pressure: pressure,
            bow_string: 0,
            drone: false,
        };
        print!("speed {speed:.1} pressure {pressure:.1} |");
        let mut last = Vec::new();
        for _window in 0..6 {
            let n = (0.5 * SR) as usize;
            let buf: Vec<f32> = (0..n).map(|_| bank.tick(&p)).collect();
            let rms = (buf.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
            print!(" {rms:.2}");
            last = buf;
        }
        // Which mode did it lock onto? Search all lags, not just near
        // the fundamental — surface sound whistles at a harmonic.
        let ac = |lag: usize| -> f32 {
            let n = 8192.min(last.len() - lag);
            (0..n).map(|i| last[i] * last[i + lag]).sum::<f32>() / n as f32
        };
        // Octave guard: for a periodic signal ac(2T) ≈ ac(T), so take
        // the smallest lag close to the maximum, not the global argmax.
        let max = (40..=420).map(&ac).fold(f32::MIN, f32::max);
        let best = (40..=420).find(|&l| ac(l) > 0.9 * max).unwrap();
        println!("  locked at ~{:.0} Hz", SR / best as f32);
    }

    // Dynamics and spectrum vs bow speed: loudness should keep scaling
    // with speed, and the harmonic rolloff should stay saw-like rather
    // than flattening into limiter crunch.
    println!("\nspeed sweep at pressure 0.5 (steady state):");
    for speed in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2] {
        let mut bank = Bank::new(&[220.0]);
        let p = Params {
            decay: 0.996,
            damping: 0.5,
            stiffness: 0.0,
            level: 1.0,
            bow_on: true,
            couple: 0.0,
            bow_speed: speed,
            bow_pressure: 0.5,
            bow_string: 0,
            drone: false,
        };
        let n = (3.0 * SR) as usize;
        let buf: Vec<f32> = (0..n).map(|_| bank.tick(&p)).collect();
        let tail = &buf[n - SR as usize..];
        let rms = (tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
        let peak = tail.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let f = measured_freq(tail, 220.0);
        print!("speed {speed:.1} | rms {rms:.2} peak {peak:.2} f {f:6.1} | harmonics dB:");
        let h1 = harmonic(tail, f, 1);
        for k in 1..=7 {
            print!(" {:5.1}", 20.0 * (harmonic(tail, f, k) / h1).log10());
        }
        println!();
    }

    // Pluck vs bow on the same string: the attack rightly differs (a
    // pluck is a broadband burst, a bow a sustained saw), but pitch,
    // decay rate and level are the string's own and must agree.
    println!("\npluck vs bow-release (rms per 0.25s):");
    let quiet = Params {
        decay: 0.996,
        damping: 0.5,
        stiffness: 0.0,
        level: 1.0,
        couple: 0.0,
        bow_on: false,
        bow_speed: 0.0,
        bow_pressure: 0.0,
        bow_string: 0,
        drone: false,
    };
    let mut bank = Bank::new(&[220.0]);
    bank.pluck(0, 0.2, &mut timber::util::Rng(7));
    let n = (2.0 * SR) as usize;
    let buf: Vec<f32> = (0..n).map(|_| bank.tick(&quiet)).collect();
    print!("pluck       |");
    for w in buf.chunks((0.25 * SR) as usize) {
        print!(
            " {:.3}",
            (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt()
        );
    }
    let peak = buf.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    println!(
        "  peak {:.2} f {:.1}",
        peak,
        measured_freq(&buf[2205..44100], 220.0)
    );

    let mut bank = Bank::new(&[220.0]);
    let bowing = Params {
        bow_on: true,
        bow_speed: 0.6,
        bow_pressure: 0.5,
        ..quiet
    };
    for _ in 0..(2.0 * SR) as usize {
        bank.tick(&bowing);
    }
    let tail: Vec<f32> = (0..n).map(|_| bank.tick(&quiet)).collect();
    print!("bow release |");
    for w in tail.chunks((0.25 * SR) as usize) {
        print!(
            " {:.3}",
            (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt()
        );
    }
    println!();
}

/// Magnitude of the k-th harmonic of fundamental `f` in `buf` (Goertzel
/// by direct correlation).
fn harmonic(buf: &[f32], f: f32, k: usize) -> f32 {
    let w = std::f32::consts::TAU * f * k as f32 / SR;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (n, s) in buf.iter().enumerate() {
        re += s * (w * n as f32).cos();
        im += s * (w * n as f32).sin();
    }
    (re * re + im * im).sqrt().max(1e-9)
}
