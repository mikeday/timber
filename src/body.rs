//! Body resonance — the missing box.
//!
//! Everything so far radiates the raw string or membrane into the void.
//! Real instruments radiate through a *body*: an air cavity and wood
//! plates with resonances of their own, which is where much of the
//! warmth and identity lives. In keeping with the house style, a body
//! here is simply another list of modes — fixed resonant filters the
//! instrument's signal passes through continuously, rather than modes
//! that are struck.
//!
//! The mode lists are plausible caricatures, not measurements: a guitar
//! box with its ~100 Hz air resonance and wood modes, a violin with its
//! A0 and the broad bridge hill, a drum shell, and a "plate" whose
//! dense long-ringing modes act as a small metallic reverb.

use crate::util::SR;
use std::f32::consts::{PI, TAU};

/// (center frequency Hz, bandwidth Hz, gain). Narrow bandwidth = long
/// ring: a two-pole's T60 is roughly 2.2/bw seconds.
type BodyMode = (f32, f32, f32);

pub const PRESETS: &[(&str, &[BodyMode])] = &[
    ("none", &[]),
    (
        "guitar",
        &[
            (98.0, 14.0, 1.0), // air (Helmholtz) resonance
            (196.0, 20.0, 0.55),
            (292.0, 28.0, 0.45),
            (399.0, 40.0, 0.35),
            (560.0, 60.0, 0.30),
            (880.0, 90.0, 0.22),
            (1300.0, 140.0, 0.15),
            (2400.0, 300.0, 0.10),
        ],
    ),
    (
        "violin",
        &[
            (275.0, 22.0, 0.9),  // A0 air
            (450.0, 28.0, 0.85), // main wood
            (550.0, 32.0, 0.7),
            (700.0, 45.0, 0.45),
            (980.0, 90.0, 0.35),
            (1400.0, 150.0, 0.30),
            (2500.0, 350.0, 0.50), // bridge hill
            (3400.0, 450.0, 0.25),
        ],
    ),
    (
        "shell",
        &[
            (60.0, 15.0, 0.8),
            (120.0, 25.0, 0.6),
            (240.0, 45.0, 0.4),
            (480.0, 80.0, 0.3),
            (900.0, 150.0, 0.25),
            (1800.0, 300.0, 0.2),
        ],
    ),
    (
        "plate",
        &[
            (157.0, 3.0, 0.30),
            (233.0, 3.5, 0.30),
            (347.0, 4.0, 0.30),
            (521.0, 4.0, 0.28),
            (779.0, 5.0, 0.26),
            (1163.0, 5.0, 0.24),
            (1741.0, 6.0, 0.22),
            (2609.0, 7.0, 0.20),
            (3907.0, 8.0, 0.18),
            (5851.0, 10.0, 0.15),
        ],
    ),
];

struct Res {
    b1: f32,
    b2: f32,
    g: f32,
    /// State-kick scale for knock(): starting the filter at y1 = kick
    /// rings at ~the mode's gain. (An input impulse won't do: these are
    /// normalized for steady-state gain, so their impulse response is
    /// nearly silent — a resonator needs ~Q cycles of drive to build.)
    kick: f32,
    y1: f32,
    y2: f32,
}

pub struct Body {
    res: Vec<Res>,
}

impl Body {
    pub fn new(modes: &[BodyMode]) -> Self {
        let res = modes
            .iter()
            .map(|&(f, bw, gain)| {
                let r = (-PI * bw / SR).exp();
                let th = TAU * f / SR;
                let b1 = 2.0 * r * th.cos();
                let b2 = r * r;
                // Exact peak normalization: evaluate |denominator| at the
                // mode's own frequency, so peak gain is `gain` for every
                // mode. (The common (1-r) shortcut leaves low modes ~8×
                // louder than high ones — peak gain goes as 1/sin θ.)
                let re = 1.0 - b1 * th.cos() + b2 * (2.0 * th).cos();
                let im = b1 * th.sin() - b2 * (2.0 * th).sin();
                Res {
                    b1,
                    b2,
                    g: gain * (re * re + im * im).sqrt(),
                    kick: gain * th.sin(),
                    y1: 0.0,
                    y2: 0.0,
                }
            })
            .collect();
        Body { res }
    }

    /// A knuckle on the box: set every mode ringing at once.
    pub fn knock(&mut self, strength: f32) {
        for r in &mut self.res {
            r.y1 += strength * r.kick;
        }
    }

    /// Wet output only; the caller mixes it with the dry signal.
    pub fn tick(&mut self, x: f32) -> f32 {
        let mut sum = 0.0;
        for r in &mut self.res {
            let y = r.g * x + r.b1 * r.y1 - r.b2 * r.y2;
            r.y2 = r.y1;
            r.y1 = y;
            sum += y;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy_near(buf: &[f32], f: f32) -> f32 {
        let w = TAU * f / SR;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (n, s) in buf.iter().enumerate() {
            re += s * (w * n as f32).cos();
            im += s * (w * n as f32).sin();
        }
        (re * re + im * im).sqrt().max(1e-9)
    }

    /// Steady-state gain at frequency f: drive with a sine, skip the
    /// transient, read the output level.
    fn response_at(modes: &[BodyMode], f: f32) -> f32 {
        let mut b = Body::new(modes);
        let mut acc = 0.0;
        let n = 22050;
        for i in 0..n {
            let y = b.tick((TAU * f * i as f32 / SR).sin());
            if i >= 4410 {
                acc += y * y;
            }
        }
        (acc / (n - 4410) as f32).sqrt()
    }

    #[test]
    fn body_colors_the_spectrum_and_stays_stable() {
        // The violin body must favor its main wood mode (450 Hz) over
        // the valley between modes (180 Hz).
        let modes = PRESETS.iter().find(|(n, _)| *n == "violin").unwrap().1;
        // Note the modest ratio: below a resonator cluster, every
        // mode's low-frequency skirt adds coherently (near-zero phase),
        // so a two-pole bank's valleys are inherently shallow — a few
        // dB of ripple, not the notches of a measured body.
        let on_mode = response_at(modes, 450.0);
        let off_mode = response_at(modes, 180.0).max(1e-9);
        assert!(
            on_mode > off_mode * 1.6,
            "no resonance shape: on {on_mode} off {off_mode}"
        );
        // And a knock must be audible, ring out, and not build up.
        let mut b = Body::new(modes);
        b.knock(0.8);
        let out: Vec<f32> = (0..44100).map(|_| b.tick(0.0)).collect();
        assert!(out.iter().all(|s| s.is_finite()));
        let early = (out[..4410].iter().map(|s| s * s).sum::<f32>() / 4410.0).sqrt();
        assert!(early > 0.05, "knock inaudible: rms {early}");
        let tail: f32 = out[33075..].iter().map(|s| s.abs()).sum();
        assert!(tail < 0.1, "body rings forever: {tail}");
    }
}
