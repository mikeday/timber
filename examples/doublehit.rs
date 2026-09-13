//! Diagnostic: a hit followed by a hard hit on the ringing modal
//! cymbal — does the ring stay a cymbal or dissolve into static?
//! Spectral flatness (geometric/arithmetic mean of the magnitude
//! spectrum) near 1 = noise; a ringing plate sits well below.
//!
//!     cargo run --release --example doublehit
use std::sync::Arc;
use timber::analyze;
use timber::cymbal::{self, Cymbal, CymbalParams, Modes};
use timber::util::{Rng, SR};
fn main() {
    let modes = Arc::new(Modes::compute());
    let (_, crash) = cymbal::default_kit()
        .into_iter()
        .find(|(n, _)| *n == "crash")
        .unwrap();
    for (nonlin, second) in [
        (0.0, 1.6),
        (4000.0, 0.0),
        (4000.0, 1.6),
        (4000.0, 3.0),
        (16000.0, 1.6),
    ] {
        let mut m = Cymbal::new(
            modes.clone(),
            CymbalParams {
                nonlin,
                level: 1.0,
                ..crash
            },
        );
        let mut rng = Rng(3);
        let mut out = Vec::new();
        m.strike(0.8);
        for _ in 0..(0.3 * SR) as usize {
            out.push(m.tick(&mut rng));
        }
        if second > 0.0 {
            m.strike(second);
        }
        for _ in 0..(1.7 * SR) as usize {
            out.push(m.tick(&mut rng));
        }
        let stat = |t0: f32, t1: f32| {
            let w = analyze::win(&out, t0, t1);
            let (mut f, mut mags) = (100.0f32, Vec::new());
            while f < 7000.0 {
                mags.push(analyze::mag(w, f).max(1e-9));
                f *= 1.02;
            }
            let am = mags.iter().sum::<f32>() / mags.len() as f32;
            let gm = (mags.iter().map(|m| m.ln()).sum::<f32>() / mags.len() as f32).exp();
            (analyze::centroid(w), gm / am, analyze::rms(w))
        };
        let (c1, fl1, r1) = stat(0.1, 0.3);
        let (c2, fl2, r2) = stat(0.5, 1.0);
        let (c3, fl3, r3) = stat(1.5, 2.0);
        println!(
            "nonlin {nonlin:>6.0} second {second:.1} | before: {c1:>5.0}Hz flat {fl1:.2} rms {r1:.3} | after 0.2-0.7s: {c2:>5.0}Hz flat {fl2:.2} rms {r2:.3} | 1.2-1.7s: {c3:>5.0}Hz flat {fl3:.2} rms {r3:.3}"
        );
    }
}
