//! Diagnostic: is the cymbal's late energy a cascade (spread over many partials) or a numerical runaway (one peak)? Prints top peaks of the 0.3–0.8 s ring and level over time.
use timber::analyze;
use timber::plate::{self, Plate, PlateParams};
use timber::util::{Rng, SR};
fn main() {
    let (_, crash) = plate::default_kit()
        .into_iter()
        .find(|(n, _)| *n == "crash")
        .unwrap();
    for (label, hf) in [("hf 0.15", 0.15), ("hf 1.0", 1.0)] {
        for strength in [1.0, 1.6, 3.0] {
            let p = PlateParams {
                hf_damp: hf,
                level: 1.0,
                ..crash
            };
            let mut m = Plate::new(p);
            let mut rng = Rng(3);
            m.strike(strength);
            let out: Vec<f32> = (0..(2.0 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
            let late = analyze::win(&out, 0.3, 0.8);
            // top 3 peaks 100 Hz..10 kHz on a 2% grid, and how much of the magnitude sum the top one is
            let mut f = 100.0;
            let mut bands = Vec::new();
            while f < 10000.0 {
                bands.push((f, analyze::mag(late, f)));
                f *= 1.02;
            }
            let total: f32 = bands.iter().map(|b| b.1).sum();
            let mut pk: Vec<(f32, f32)> = (1..bands.len() - 1)
                .filter(|&i| bands[i].1 > bands[i - 1].1 && bands[i].1 >= bands[i + 1].1)
                .map(|i| bands[i])
                .collect();
            pk.sort_by(|a, b| b.1.total_cmp(&a.1));
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            print!(
                "{label:8} str {strength:.1} | peak {peak:.2} | centroid 20-50 {:>5.0} 100-200 {:>5.0} 300-800 {:>5.0} | rms 0.3-0.8s {:.4} 1.5-2s {:.4} | top:",
                analyze::centroid(analyze::win(&out, 0.02, 0.05)),
                analyze::centroid(analyze::win(&out, 0.1, 0.2)),
                analyze::centroid(late),
                analyze::rms(late),
                analyze::rms(analyze::win(&out, 1.5, 2.0))
            );
            for (f, m) in pk.iter().take(3) {
                print!(" {:.0}Hz({:.0}%)", f, 100.0 * m / total);
            }
            println!();
        }
    }
}
