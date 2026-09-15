//! Diagnostic: a modal drum against its mesh twin, side by side —
//! envelope, attack brightness over time, pitch trajectory, and a mode
//! table with per-mode decay. Same fundamental, same hit strength, both
//! peak-normalized, so what differs is the mechanism.
//!
//!     cargo run --release --example compare [kick|snare|tomhi|tomlo]

use timber::analyze;
use timber::drums;
use timber::mesh::{self, Mesh, MeshParams};
use timber::modal::{MEMBRANE, MEMBRANE_CENTER};
use timber::util::{Rng, SR};

/// Which drum: the modal pad (as the desk defines it) against the
/// library's mesh default for it, retuned to the modal fundamental.
struct Pair {
    name: &'static str,
    modal: drums::PadParams,
    mesh: MeshParams,
}

fn pairs() -> Vec<Pair> {
    let modal = drums::PadParams {
        modes: MEMBRANE,
        freq: 110.0,
        glide: 0.0,
        glide_time: 0.1,
        damp: 1.0,
        noise: 0.0,
        noise_decay: 0.1,
        noise_tone: 0.0,
        bloom: 0.0,
        bloom_delay: 0.025,
        bloom_spread: 0.0,
        drive: 0.0,
        shimmer: 0.0,
        attack: 0.0,
        claps: 1,
        tremolo: 0.0,
        bloom_lo: 0.0,
        bleed: 0.0,
        roll_rate: 14.0,
        roll_strength: 0.75,
        damper: false,
        level: 1.0,
        choke: None,
    };
    let kit = mesh::default_kit();
    let mesh_pad = |name: &str, freq: f32| {
        let (_, p) = kit.iter().find(|(n, _)| *n == name).expect("pad");
        MeshParams {
            freq,
            level: 1.0,
            ..*p
        }
    };
    vec![
        Pair {
            name: "kick",
            modal: drums::PadParams {
                modes: MEMBRANE_CENTER,
                freq: 60.0,
                glide: 0.5,
                glide_time: 0.035,
                damp: 0.35,
                noise: 0.18,
                noise_decay: 0.004,
                noise_tone: 0.0,
                bloom: 0.0,
                bloom_delay: 0.025,
                bloom_spread: 0.0,
                drive: 1.8,
                ..modal
            },
            mesh: mesh_pad("kick", 60.0),
        },
        Pair {
            name: "snare",
            modal: drums::PadParams {
                freq: 185.0,
                glide: 0.15,
                glide_time: 0.05,
                noise: 0.9,
                noise_decay: 0.09,
                ..modal
            },
            mesh: mesh_pad("snare", 185.0),
        },
        Pair {
            name: "tomhi",
            modal: drums::PadParams {
                freq: 155.0,
                glide: 0.25,
                glide_time: 0.12,
                ..modal
            },
            mesh: mesh_pad("tom hi", 155.0),
        },
        Pair {
            name: "tomlo",
            modal: drums::PadParams {
                freq: 80.0,
                glide: 0.25,
                glide_time: 0.12,
                ..modal
            },
            mesh: mesh_pad("tom lo", 80.0),
        },
    ]
}

const SECS: f32 = 1.5;

fn modal_render(p: drums::PadParams) -> Vec<f32> {
    let mut kit = drums::Kit::new(vec![p]);
    let mut rng = Rng(3);
    kit.strike(0);
    (0..(SECS * SR) as usize)
        .map(|_| kit.tick(&mut rng))
        .collect()
}

fn mesh_render(p: MeshParams) -> Vec<f32> {
    let mut m = Mesh::new(p);
    let mut rng = Rng(3);
    m.strike(1.0);
    (0..(SECS * SR) as usize)
        .map(|_| m.tick(&mut rng))
        .collect()
}

fn main() {
    let want = std::env::args().nth(1).unwrap_or_else(|| "tomhi".into());
    let pair = pairs()
        .into_iter()
        .find(|p| p.name == want)
        .expect("drum: kick | snare | tomhi | tomlo");
    let f0 = Some(pair.mesh.freq);
    println!(
        "{}: modal vs mesh (and mesh with tension 0), all at {} Hz, strength 1.0, peak-normalized\n",
        pair.name, pair.mesh.freq
    );
    let reports = [
        analyze::report("modal", &analyze::normalize(modal_render(pair.modal)), f0),
        analyze::report("mesh", &analyze::normalize(mesh_render(pair.mesh)), f0),
        analyze::report(
            "mesh t0",
            &analyze::normalize(mesh_render(MeshParams {
                tension: 0.0,
                ..pair.mesh
            })),
            f0,
        ),
    ];
    analyze::print_side_by_side(&reports);
}
