//! Diagnostic: measure recorded drum hits — real drums, for calibrating
//! the mesh kit — with the same report `compare` prints for our own.
//! Any WAV (mono or multichannel, any rate); leading silence is
//! trimmed, the hit peak-normalized. Several files print side by side.
//!
//!     cargo run --release --example analyze -- kick.wav [tom.wav ...]
//!     cargo run --release --example analyze -- --f0 58 kick.wav
//!     cargo run --release --example analyze -- samples/floortom.wav mesh:tomlo
//!
//! --f0 pins the fundamental when the automatic pick (strongest
//! component under 400 Hz in the settled ring) lands on an overtone.
//! `mesh:<kick|snare|tomhi|tom|tomlo>` or `plate:<ride|crash>` (FD plate) or `cymbal:<ride|crash>` (modal) renders that mesh default (strength
//! 1.0, or --strength S) as a column, for calibrating against the
//! recording beside it.

use timber::analyze::{self, Report};
use timber::cymbal::{self, Cymbal};
use timber::mesh::{self, Mesh, MeshParams};
use timber::plate::{self, Plate, PlateParams};
use timber::util::{Rng, SR};

fn load(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let full = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / full)
                .collect()
        }
    };
    // Mono mix.
    let mono: Vec<f32> = samples
        .chunks(ch)
        .map(|fr| fr.iter().sum::<f32>() / ch as f32)
        .collect();
    // Resample to our rate (linear: fine for analysis of a drum hit).
    let step = spec.sample_rate as f32 / SR;
    let n = (mono.len() as f32 / step) as usize;
    (0..n)
        .map(|i| {
            let x = i as f32 * step;
            let j = x as usize;
            let t = x - j as f32;
            let a = mono[j];
            let b = mono[(j + 1).min(mono.len() - 1)];
            a + (b - a) * t
        })
        .collect()
}

fn render_mesh(name: &str, strength: f32) -> Vec<f32> {
    let key = name.replace(' ', "");
    let (_, p) = mesh::default_kit()
        .into_iter()
        .find(|(n, _)| n.replace(' ', "") == key)
        .unwrap_or_else(|| panic!("no mesh pad '{name}'"));
    let mut m = Mesh::new(MeshParams { level: 1.0, ..p });
    let mut rng = Rng(3);
    m.strike(strength);
    (0..(1.5 * SR) as usize).map(|_| m.tick(&mut rng)).collect()
}

fn render_plate(name: &str, strength: f32) -> Vec<f32> {
    let (_, p) = plate::default_kit()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no plate '{name}'"));
    let mut m = Plate::new(PlateParams { level: 1.0, ..p });
    let mut rng = Rng(3);
    m.strike(strength);
    (0..(1.5 * SR) as usize).map(|_| m.tick(&mut rng)).collect()
}

fn render_cymbal(name: &str, strength: f32) -> Vec<f32> {
    let (_, p) = cymbal::default_kit()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no cymbal '{name}'"));
    let modes = std::sync::Arc::new(cymbal::Modes::compute());
    let mut m = Cymbal::new(modes, PlateParams { level: 1.0, ..p });
    let mut rng = Rng(3);
    m.strike(strength);
    (0..(1.5 * SR) as usize).map(|_| m.tick(&mut rng)).collect()
}

fn main() {
    let mut f0: Option<f32> = None;
    let mut strength = 1.0f32;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--f0" {
            f0 = args.next().and_then(|v| v.parse().ok());
        } else if a == "--strength" {
            strength = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        } else {
            files.push(a);
        }
    }
    if files.is_empty() {
        eprintln!("usage: analyze [--f0 HZ] file.wav [more.wav ...]");
        std::process::exit(2);
    }
    let reports: Vec<Report> = files
        .iter()
        .map(|path| {
            if let Some(name) = path.strip_prefix("mesh:") {
                let hit = analyze::normalize(render_mesh(name, strength));
                return analyze::report(&format!("mesh:{name}"), &hit, f0);
            }
            if let Some(name) = path.strip_prefix("cymbal:") {
                let hit = analyze::normalize(render_cymbal(name, strength));
                return analyze::report(&format!("cymbal:{name}"), &hit, f0);
            }
            if let Some(name) = path.strip_prefix("plate:") {
                let hit = analyze::normalize(render_plate(name, strength));
                return analyze::report(&format!("plate:{name}"), &hit, f0);
            }
            let raw = load(path);
            let mut hit = analyze::normalize(analyze::trim_onset(&raw).to_vec());
            // Pad so every window exists even for a short sample.
            hit.resize(hit.len().max((1.5 * SR) as usize), 0.0);
            let name = std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone());
            let name: String = name.chars().take(9).collect();
            analyze::report(&name, &hit, f0)
        })
        .collect();
    analyze::print_side_by_side(&reports);
    for r in &reports {
        println!("\n[{}]", r.name);
        analyze::suggest(r);
    }
}
