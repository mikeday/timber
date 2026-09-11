//! Model 3b: streaming voice — the formant filter as a held instrument.
//!
//! voice.rs renders finished syllables. The mouth instead phonates for
//! as long as a gate is held, with the first two formants steered live:
//! the classic F1/F2 vowel plane, where F1 is jaw openness and F2 is
//! tongue position, and every vowel is a point. Drag through it and the
//! voice morphs ee → eh → ah → oh → oo continuously — vowels as a
//! *space*, not a list.

use crate::util::{AtomicF32, Rng, SR};
use crate::voice::Glottis;
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// The F1/F2 ranges of the vowel plane (log-mapped by the UI).
pub const F1_RANGE: (f32, f32) = (250.0, 850.0);
pub const F2_RANGE: (f32, f32) = (700.0, 2400.0);

pub struct Ctl {
    pub freq: AtomicF32,
    pub f1: AtomicF32,
    pub f2: AtomicF32,
    pub breath: AtomicF32,
    pub vibrato: AtomicF32,
    pub level: AtomicF32,
    /// Held = phonating. The envelope lives audio-side.
    pub gate: AtomicBool,
}

impl Ctl {
    pub fn new() -> Self {
        Ctl {
            freq: AtomicF32::new(165.0),
            f1: AtomicF32::new(730.0),
            f2: AtomicF32::new(1090.0),
            breath: AtomicF32::new(0.06),
            vibrato: AtomicF32::new(0.012),
            level: AtomicF32::new(0.8),
            gate: AtomicBool::new(false),
        }
    }
}

impl Default for Ctl {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct Params {
    pub freq: f32,
    pub f1: f32,
    pub f2: f32,
    pub breath: f32,
    pub vibrato: f32,
    pub level: f32,
    pub gate: bool,
}

impl Params {
    pub fn read(ctl: &Ctl) -> Self {
        Params {
            freq: ctl.freq.get(),
            f1: ctl.f1.get(),
            f2: ctl.f2.get(),
            breath: ctl.breath.get(),
            vibrato: ctl.vibrato.get(),
            level: ctl.level.get(),
            gate: ctl.gate.load(Relaxed),
        }
    }
}

/// Formant smoothing (~11 ms), the same feel as the other models.
const SMOOTH: f32 = 0.002;

pub struct Mouth {
    /// The shared humanized source (voice::Glottis) — one larynx
    /// implementation for both streaming engines.
    g: Glottis,
    f1_s: f32,
    f2_s: f32,
    res: [(f32, f32); 4],
}

impl Mouth {
    pub fn new() -> Self {
        Mouth {
            g: Glottis::new(165.0),
            f1_s: 730.0,
            f2_s: 1090.0,
            res: [(0.0, 0.0); 4],
        }
    }

    pub fn tick(&mut self, p: &Params, rng: &mut Rng) -> f32 {
        let g = self.g.tick(p.freq, p.vibrato, p.gate, rng);
        if self.g.amp < 1e-4 && !p.gate {
            return 0.0;
        }

        self.f1_s += SMOOTH * (p.f1 - self.f1_s);
        self.f2_s += SMOOTH * (p.f2 - self.f2_s);

        // Aspiration pulses with the glottal cycle — air rushes while
        // the fold is open, not as a continuous hiss.
        let src = g * (1.0 - p.breath) + rng.next() * p.breath * (0.3 + 0.7 * g);

        // Four formants: the two steered ones, and fixed F3/F4 for
        // presence. Same resonator as voice.rs.
        let formants = [
            (self.f1_s, 90.0, 1.0),
            (self.f2_s, 110.0, 0.6),
            (2500.0, 150.0, 0.25),
            (3300.0, 200.0, 0.1),
        ];
        let mut sample = 0.0;
        for (k, (fc, bw, gain)) in formants.iter().enumerate() {
            let r = (-std::f32::consts::PI * bw / SR).exp();
            let (y1, y2) = self.res[k];
            let y = (1.0 - r) * src + 2.0 * r * (TAU * fc / SR).cos() * y1 - r * r * y2;
            self.res[k] = (y, y1);
            sample += gain * y;
        }
        sample * self.g.amp * p.level
    }
}

impl Default for Mouth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::measured_freq;

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    #[test]
    fn gated_vowel_sustains_then_releases() {
        let mut m = Mouth::new();
        let mut rng = Rng(5);
        let on = Params {
            freq: 165.0,
            f1: 730.0,
            f2: 1090.0,
            breath: 0.06,
            vibrato: 0.0,
            level: 0.8,
            gate: true,
        };
        let out: Vec<f32> = (0..44100).map(|_| m.tick(&on, &mut rng)).collect();
        assert!(out.iter().all(|s| s.is_finite()));
        let held = rms(&out[22050..]);
        assert!(held > 0.02, "mouth too quiet: {held}");
        let f = measured_freq(&out[8820..], 165.0);
        let cents = 1200.0 * (f / 165.0).log2();
        assert!(
            cents.abs() < 30.0,
            "mouth pitch off: {f:.2} Hz ({cents:+.1}c)"
        );

        let off = Params { gate: false, ..on };
        let tail: Vec<f32> = (0..22050).map(|_| m.tick(&off, &mut rng)).collect();
        assert!(
            rms(&tail[11025..]) < held * 0.05,
            "mouth kept sounding after release"
        );
    }

    #[test]
    fn formants_move_the_spectrum() {
        // "ee" must put far more energy near 2290 Hz than "ah" does.
        let energy_near = |buf: &[f32], f: f32| {
            let w = TAU * f / SR;
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, s) in buf.iter().enumerate() {
                re += s * (w * n as f32).cos();
                im += s * (w * n as f32).sin();
            }
            (re * re + im * im).sqrt()
        };
        let sing = |f1: f32, f2: f32| {
            let mut m = Mouth::new();
            let mut rng = Rng(5);
            let p = Params {
                freq: 165.0,
                f1,
                f2,
                breath: 0.06,
                vibrato: 0.0,
                level: 0.8,
                gate: true,
            };
            let out: Vec<f32> = (0..44100).map(|_| m.tick(&p, &mut rng)).collect();
            out[22050..].to_vec()
        };
        let ee = sing(270.0, 2290.0);
        let ah = sing(730.0, 1090.0);
        let ratio = (energy_near(&ee, 2310.0) / energy_near(&ah, 2310.0)).max(1e-6);
        assert!(
            ratio > 3.0,
            "ee/ah energy ratio near F2(ee) only {ratio:.2}"
        );
    }
}
