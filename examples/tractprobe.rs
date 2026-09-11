//! Diagnostic: map the Kelly-Lochbaum tract's articulation space.
//! Drives the tube with pure breath noise (so the output spectrum is
//! the tract's own resonances) and reports the first formant peaks per
//! (tongue position, constriction, lips) — the data from which the
//! vowel presets are calibrated.
//!
//!     cargo run --release --example tractprobe

use timber::tract::{Params, Tract};
use timber::util::{Rng, SR};

fn spectrum_peaks(buf: &[f32]) -> Vec<(f32, f32)> {
    // Band magnitudes every 25 Hz, lightly smoothed, then local maxima.
    let bands: Vec<(f32, f32)> = (6..130)
        .map(|k| {
            let f = k as f32 * 25.0;
            let w = std::f32::consts::TAU * f / SR;
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, s) in buf.iter().enumerate() {
                re += s * (w * n as f32).cos();
                im += s * (w * n as f32).sin();
            }
            // Whiten against the aspiration lowpass and radiation tilt
            // (see tract tests).
            let a = 0.89f32;
            let hlp = (0.11 * 0.11) / (1.0 - 2.0 * a * w.cos() + a * a);
            let b = 0.32f32;
            let hrad = (1.0 - 2.0 * 0.95 * w.cos() + 0.95 * 0.95).sqrt() * 0.68
                / (1.0 - 2.0 * b * w.cos() + b * b).sqrt();
            (f, (re * re + im * im).sqrt() / (hlp * hrad))
        })
        .collect();
    let sm: Vec<f32> = (0..bands.len())
        .map(|i| {
            let lo = i.saturating_sub(1);
            let hi = (i + 2).min(bands.len());
            bands[lo..hi].iter().map(|b| b.1).sum::<f32>() / (hi - lo) as f32
        })
        .collect();
    let mut peaks = Vec::new();
    for i in 2..sm.len() - 2 {
        if sm[i] > sm[i - 1] && sm[i] >= sm[i + 1] && sm[i] > sm[i - 2] && sm[i] > sm[i + 2] {
            peaks.push((bands[i].0, sm[i]));
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    peaks.truncate(5);
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    peaks
}

pub fn formants(tongue_pos: f32, constrict: f32, lips: f32) -> Vec<(f32, f32)> {
    let mut tract = Tract::new();
    let mut rng = Rng(11);
    let p = Params {
        freq: 120.0,
        tongue_pos,
        constrict,
        lips,
        velum: 0.0,
        tip: 0.0,
        voiced: 0.0,
        breath: 1.0, // pure noise: the spectrum is the tube's response
        flow: 1.0,
        vibrato: 0.0,
        level: 1.0,
        gate: true,
    };
    let n = (1.5 * SR) as usize;
    let buf: Vec<f32> = (0..n).map(|_| tract.tick(&p, &mut rng)).collect();
    spectrum_peaks(&buf[n - SR as usize..])
}

fn main() {
    println!("calibrated vowels:");
    for (name, tp, con, lips) in timber::tract::VOWELS {
        print!("{name} ({tp:.2},{con:.2},{lips:.2}) |");
        for (f, m) in formants(tp, con, lips) {
            print!("  {f:4.0}Hz({m:6.1})");
        }
        println!();
    }
    for lips in [0.15, 0.5, 1.0] {
        println!("--- lips {lips:.2}");
        for tp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            for con in [0.3, 0.6, 0.85] {
                let peaks = formants(tp, con, lips);
                print!("tongue {tp:.1} constrict {con:.2} |");
                for (f, m) in &peaks {
                    print!("  {f:4.0}Hz({m:5.1})");
                }
                println!();
            }
        }
    }
}
