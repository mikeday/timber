//! Diagnostic: the modal cymbal's wash against its coupling budget —
//! for several (modes, stress modes, keep threshold) budgets, the
//! coupling term count, cost, and the brightness climb of a hard hit
//! (centroid per window, linear plate first for reference).
//!
//!     cargo run --release --example modalprobe
use std::sync::Arc;
use timber::analyze;
use timber::cymbal::{self, Cymbal, CymbalParams, Modes};
use timber::util::{Rng, SR};
fn main() {
    let (_, crash) = cymbal::default_kit()
        .into_iter()
        .find(|(n, _)| *n == "crash")
        .unwrap();
    println!(
        "{:>5} {:>6} {:>5} | {:>6} {:>6} | {:>6} | {:>5} {:>5} {:>6} {:>7} {:>7} | {:>5} {:>5}",
        "modes",
        "stress",
        "keep",
        "cross",
        "self",
        "ms/2s",
        "5-20",
        "20-50",
        "50-100",
        "100-200",
        "200-400",
        "peak",
        "flat"
    );
    for (nm, ns, keep) in [
        (120, 20, 0.15),
        (120, 8, 0.05),
        (120, 8, 0.02),
        (120, 4, 0.01),
        (160, 6, 0.02),
        (200, 4, 0.02),
    ] {
        let modes = Arc::new(Modes::compute_with(nm, ns, keep));
        let (cross, diag) = modes.coupling_stats();
        for nonlin in [0.0, 4000.0, 16000.0] {
            let mut m = Cymbal::new(
                modes.clone(),
                CymbalParams {
                    nonlin,
                    level: 1.0,
                    ..crash
                },
            );
            let mut rng = Rng(3);
            let t = std::time::Instant::now();
            m.strike(1.6);
            let out: Vec<f32> = (0..(2.0 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
            let ms = t.elapsed().as_millis();
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            let w = analyze::win(&out, 0.5, 1.0);
            let (mut f, mut mags) = (100.0f32, Vec::new());
            while f < 7000.0 {
                mags.push(analyze::mag(w, f).max(1e-9));
                f *= 1.02;
            }
            let am = mags.iter().sum::<f32>() / mags.len() as f32;
            let gm = (mags.iter().map(|m| m.ln()).sum::<f32>() / mags.len() as f32).exp();
            print!("{nm:>5} {ns:>6} {keep:>5.2} | {cross:>6} {diag:>6} | {ms:>6} |");
            for &(a, b) in &analyze::WINDOWS[1..6] {
                print!(" {:>6.0}", analyze::centroid(analyze::win(&out, a, b)));
            }
            println!(" | {peak:>5.2} {:>5.2}   (nonlin {nonlin:.0})", gm / am);
        }
    }
}
