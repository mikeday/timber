//! Diagnostic: the cymbal's wash — does energy climb into the high
//! partials *after* the hit? Brightness (spectral centroid) per window
//! and level, across the nonlinear coupling and strike strength.
//!
//!     cargo run --release --example washprobe

use timber::analyze;
use timber::plate::{self, Plate, PlateParams};
use timber::util::{Rng, SR};

fn main() {
    let (_, crash) = plate::default_kit()
        .into_iter()
        .find(|(n, _)| *n == "crash")
        .unwrap();
    println!("crash: centroid (Hz) per window, then peak and 1 s rms");
    println!(
        "  {:>6} {:>5} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} | {:>5} {:>6} {:>5}",
        "nonlin",
        "str",
        "0-5",
        "5-20",
        "20-50",
        "50-100",
        "100-200",
        "200-400",
        "peak",
        "rms",
        "ms"
    );
    for nonlin in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0] {
        for strength in [1.0, 1.6, 3.0] {
            let p = PlateParams {
                nonlin,
                level: 1.0,
                ..crash
            };
            let mut m = Plate::new(p);
            let mut rng = Rng(3);
            let t0 = std::time::Instant::now();
            m.strike(strength);
            let out: Vec<f32> = (0..(1.5 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
            let ms = t0.elapsed().as_millis();
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            let rms = analyze::rms(&out[..44100]);
            print!("  {nonlin:>6.1} {strength:>5.1} |");
            for &(t0, t1) in &analyze::WINDOWS[..6] {
                print!(" {:>6.0}", analyze::centroid(analyze::win(&out, t0, t1)));
            }
            println!(" | {peak:>5.2} {rms:>6.3} {ms:>5}");
        }
    }
}
