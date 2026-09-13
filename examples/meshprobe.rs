//! Diagnostic: strike the mesh drums and report peak level, fundamental
//! and the strongest mode ratios — level calibration and a check that
//! the disc's modes are the membrane's.
//!
//!     cargo run --release --example meshprobe

use timber::mesh::{self, Mesh};
use timber::util::{Rng, SR};

fn main() {
    let kit = mesh::default_kit();
    for (name, p) in kit {
        for strength in [0.5, 1.0, 1.6, 3.0] {
            let mut m = Mesh::new(p);
            let mut rng = Rng(3);
            m.strike(strength);
            let out: Vec<f32> = (0..(1.2 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            let rms = (out[..22050].iter().map(|s| s * s).sum::<f32>() / 22050.0).sqrt();
            // Band scan for the strongest few peaks.
            let bands: Vec<(f32, f32)> = (1..200)
                .map(|k| {
                    let f = k as f32 * 4.0 + 20.0;
                    let w = std::f32::consts::TAU * f / SR;
                    let (mut re, mut im) = (0.0f32, 0.0f32);
                    for (n, s) in out[..44100].iter().enumerate() {
                        re += s * (w * n as f32).cos();
                        im += s * (w * n as f32).sin();
                    }
                    (f, (re * re + im * im).sqrt())
                })
                .collect();
            let mut pk: Vec<(f32, f32)> = (1..bands.len() - 1)
                .filter(|&i| bands[i].1 > bands[i - 1].1 && bands[i].1 >= bands[i + 1].1)
                .map(|i| bands[i])
                .collect();
            pk.sort_by(|a, b| b.1.total_cmp(&a.1));
            pk.truncate(4);
            pk.sort_by(|a, b| a.0.total_cmp(&b.0));
            print!("{name:7} strength {strength:.1} | peak {peak:.2} rms {rms:.3} | modes:");
            for (f, _) in &pk {
                print!(" {:.0}Hz(x{:.2})", f, f / p.freq);
            }
            println!();
        }
    }
}
