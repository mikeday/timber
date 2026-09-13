//! Diagnostic: the stick model — contact time, peak level and attack
//! brightness (spectral centroid of the first 50 ms) across stick
//! hardness and strike strength, for a felt kick and a tom.
//!
//!     cargo run --release --example stickprobe
use timber::mesh::{Mesh, MeshParams};
use timber::util::{Rng, SR};
fn main() {
    let kick = MeshParams {
        freq: 55.0,
        decay: 0.35,
        hf_damp: 0.3,
        tension: 15.0,
        strike_pos: 0.15,
        rattle: 0.0,
        click: 0.0,
        drive: 0.0,
        air: 0.0,
        reso_freq: 50.0,
        reso_decay: 0.5,
        hardness: 0.1,
        mallet: 3.0,
        level: 1.0,
    };
    let tom = MeshParams {
        freq: 165.0,
        decay: 0.5,
        hf_damp: 0.03,
        tension: 6.0,
        strike_pos: 0.5,
        rattle: 0.0,
        click: 0.0,
        drive: 0.0,
        air: 0.0,
        reso_freq: 50.0,
        reso_decay: 0.5,
        hardness: 0.7,
        mallet: 1.5,
        level: 1.0,
    };
    for (base, hardness) in [(kick, 0.1), (tom, 0.0), (tom, 0.5), (tom, 1.0)] {
        for strength in [0.5, 1.0, 1.6] {
            let p = MeshParams { hardness, ..base };
            let mut m = Mesh::new(p);
            let mut rng = Rng(3);
            m.strike(strength);
            let out: Vec<f32> = (0..(1.0 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            // Spectral centroid of the first 50 ms via band scan.
            let (mut num, mut den) = (0.0f32, 0.0f32);
            let mut f = 50.0;
            while f < 6000.0 {
                let w = std::f32::consts::TAU * f / SR;
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for (n, s) in out[..2205].iter().enumerate() {
                    re += s * (w * n as f32).cos();
                    im += s * (w * n as f32).sin();
                }
                let mag = (re * re + im * im).sqrt();
                num += f * mag;
                den += mag;
                f *= 1.05;
            }
            println!("{:.0}Hz hardness {hardness:.1}", p.freq);
            println!(
                "    strength {strength:.1} | contact {:.2} ms | peak {peak:.2} | centroid {:.0} Hz",
                m.contact_steps() as f32 / 22.05,
                num / den
            );
        }
    }
}
