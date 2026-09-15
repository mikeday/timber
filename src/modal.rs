//! Model 2: modal synthesis — drums, bells, struck things.
//!
//! An object is a list of modes: the frequencies it likes to ring at, how
//! loud each rings, and how fast each dies. A 1D string's modes are integer
//! multiples of the fundamental, which is why it sounds pitched. A 2D
//! membrane's modes fall at the zeros of Bessel functions — 1.00, 1.59,
//! 2.14, 2.30, 2.65... — inharmonic and densely packed, which is why a
//! drum sounds like a drum. Timbre is literally this list of numbers.

use crate::util::{Rng, SR, fade_out, normalize};
use std::f32::consts::TAU;

#[derive(Clone, Copy)]
pub struct Mode {
    /// Frequency as a multiple of the fundamental.
    pub ratio: f32,
    /// How strongly the strike excites this mode. Hitting a real drum in
    /// the center favors the low symmetric modes; near the rim, everything.
    pub gain: f32,
    /// Decay time constant in seconds. High modes die first.
    pub decay: f32,
}

#[derive(Clone, Copy)]
pub struct Hit<'a> {
    pub freq: f32,
    pub modes: &'a [Mode],
    pub duration: f32,
    /// Tension modulation: a hard strike stretches the membrane, so pitch
    /// starts this fraction sharp and droops as it settles. The tom sound.
    pub glide: f32,
    /// Seconds for the tension to settle back.
    pub glide_time: f32,
    /// Muffling: scales every mode's decay time. 1.0 is the bare object;
    /// 0.3 is a pillow against the head — the tight, dry studio kick.
    /// Thump is tightness: the whack-then-gone contrast, not length.
    pub damp: f32,
    /// Rattle mixed in on top — snare wires against the bottom head.
    pub noise: f32,
    pub noise_decay: f32,
    /// Saturation. A hard-driven membrane vibrates nonlinearly, folding
    /// energy into harmonics of its modes — this is most of what makes a
    /// kick sound *loud* rather than merely low, since ears (and small
    /// speakers) are weak at 50 Hz but fine at its overtones. 0 is clean;
    /// 2-4 is meaty; silly values are fuzz.
    pub drive: f32,
    /// Peak amplitude, 0.0..1.0.
    pub level: f32,
}

impl Default for Hit<'_> {
    fn default() -> Self {
        Hit {
            freq: 110.0,
            modes: MEMBRANE,
            duration: 1.5,
            glide: 0.0,
            glide_time: 0.1,
            damp: 1.0,
            noise: 0.0,
            noise_decay: 0.1,
            drive: 0.0,
            level: 0.8,
        }
    }
}

/// Ideal circular membrane, struck off-center: Bessel-function ratios.
pub const MEMBRANE: &[Mode] = &[
    Mode {
        ratio: 1.00,
        gain: 1.00,
        decay: 0.40,
    },
    Mode {
        ratio: 1.59,
        gain: 0.55,
        decay: 0.28,
    },
    Mode {
        ratio: 2.14,
        gain: 0.40,
        decay: 0.20,
    },
    Mode {
        ratio: 2.30,
        gain: 0.35,
        decay: 0.18,
    },
    Mode {
        ratio: 2.65,
        gain: 0.25,
        decay: 0.14,
    },
    Mode {
        ratio: 2.92,
        gain: 0.20,
        decay: 0.12,
    },
    Mode {
        ratio: 3.16,
        gain: 0.15,
        decay: 0.10,
    },
    Mode {
        ratio: 3.50,
        gain: 0.10,
        decay: 0.08,
    },
];

/// Only the round, symmetric modes — a strike dead center. Boomy.
pub const MEMBRANE_CENTER: &[Mode] = &[
    Mode {
        ratio: 1.00,
        gain: 1.00,
        decay: 0.40,
    },
    Mode {
        ratio: 2.30,
        gain: 0.30,
        decay: 0.15,
    },
    Mode {
        ratio: 3.60,
        gain: 0.10,
        decay: 0.08,
    },
];

/// Church-bell-ish partials: hum, prime, minor third, fifth, nominal...
/// Nothing membrane about these ratios — that's the whole point. Swap the
/// list, swap the object.
pub const BELL: &[Mode] = &[
    Mode {
        ratio: 0.50,
        gain: 0.60,
        decay: 3.5,
    },
    Mode {
        ratio: 1.00,
        gain: 1.00,
        decay: 3.0,
    },
    Mode {
        ratio: 1.19,
        gain: 0.50,
        decay: 2.2,
    },
    Mode {
        ratio: 1.50,
        gain: 0.45,
        decay: 1.8,
    },
    Mode {
        ratio: 2.00,
        gain: 0.40,
        decay: 1.4,
    },
    Mode {
        ratio: 2.74,
        gain: 0.25,
        decay: 0.9,
    },
    Mode {
        ratio: 3.00,
        gain: 0.15,
        decay: 0.7,
    },
];

/// A triangle: a solid steel rod, free at both ends, bent — and the
/// bend is mostly ergonomics. The ratios are the free-free rod's
/// bending-mode series (growing ~quadratically, far more stretched than
/// a bell); what the corners *do* contribute is coupling the two
/// bending polarizations, splitting each mode into a slightly detuned
/// doublet — and close pairs beat, which is the shimmer. The splits are
/// near-constant in absolute Hz (a few Hz of beat at any pitch):
/// proportional splits push high pairs into 20-150 Hz beating, which
/// the ear hears as roughness — buzz, not shimmer. Set the
/// detunes to zero and you're striking the straight rod.
///
/// These baked pairs deliberately overlap the runtime `shimmer` knob:
/// geometry gives each pair member its own gain and decay, which the
/// symmetric knob cannot express — so the triangle keeps its physics
/// and the knob stays a generic effect on top.
pub const TRIANGLE: &[Mode] = &[
    Mode {
        ratio: 1.000,
        gain: 0.70,
        decay: 6.0,
    },
    Mode {
        ratio: 1.003,
        gain: 0.65,
        decay: 5.5,
    },
    Mode {
        ratio: 2.756,
        gain: 1.00,
        decay: 4.5,
    },
    Mode {
        ratio: 2.760,
        gain: 0.90,
        decay: 4.2,
    },
    Mode {
        ratio: 5.404,
        gain: 0.80,
        decay: 2.8,
    },
    Mode {
        ratio: 5.409,
        gain: 0.70,
        decay: 2.6,
    },
    Mode {
        ratio: 8.933,
        gain: 0.50,
        decay: 1.6,
    },
    Mode {
        ratio: 8.938,
        gain: 0.45,
        decay: 1.5,
    },
    Mode {
        ratio: 13.34,
        gain: 0.18,
        decay: 1.0,
    },
    Mode {
        ratio: 13.345,
        gain: 0.16,
        decay: 0.95,
    },
    Mode {
        ratio: 18.64,
        gain: 0.10,
        decay: 0.6,
    },
    Mode {
        ratio: 18.645,
        gain: 0.09,
        decay: 0.55,
    },
];

pub fn render(h: &Hit, rng: &mut Rng) -> Vec<f32> {
    let len = (h.duration * SR) as usize;
    let mut out = vec![0.0f32; len];

    for m in h.modes {
        let mut phase = 0.0f32;
        for (i, s) in out.iter_mut().enumerate() {
            let t = i as f32 / SR;
            // Pitch glide: every mode rides the same tension curve.
            let f = h.freq * m.ratio * (1.0 + h.glide * (-t / h.glide_time).exp());
            phase = (phase + f / SR) % 1.0;
            *s += m.gain * (-t / (m.decay * h.damp)).exp() * (TAU * phase).sin();
        }
    }

    if h.noise > 0.0 {
        // First-difference the noise for a bright, papery rattle rather
        // than a dull roar.
        let mut prev = 0.0f32;
        for (i, s) in out.iter_mut().enumerate() {
            let t = i as f32 / SR;
            let n = rng.next();
            *s += h.noise * (n - prev) * (-t / h.noise_decay).exp();
            prev = n;
        }
    }

    if h.drive > 0.0 {
        // Normalize first so drive means the same thing at any mode-list
        // gain sum, then waveshape: tanh compresses the peaks and grows
        // odd harmonics, fattening the sound without raising its peak.
        normalize(&mut out, 1.0);
        for s in out.iter_mut() {
            *s = (h.drive * *s).tanh();
        }
    }
    normalize(&mut out, h.level);
    fade_out(&mut out);
    out
}

/// A synth cymbal: not a plate, a *picture* of one. Plate partials
/// space out roughly linearly (mode count grows ∝ frequency) with no
/// harmonic relation; the lowest partials are quiet (a cymbal's
/// weight sits at 2–6 kHz — loud lows made it a gong) and decays
/// shorten only gently upward.
/// Generated once with a seeded jitter — the point is the density
/// and the irregularity, not any particular number. The "record"
/// cymbal to the modal plate's "instrument", as the modal kick is
/// to the mesh kick; the crash's wash is the pad's noise burst plus
/// its bloom (drums.rs).
pub const CYMBAL: &[Mode] = &[
    Mode {
        ratio: 0.98,
        gain: 0.17,
        decay: 1.40,
    },
    Mode {
        ratio: 1.37,
        gain: 0.19,
        decay: 1.35,
    },
    Mode {
        ratio: 1.76,
        gain: 0.27,
        decay: 1.30,
    },
    Mode {
        ratio: 2.08,
        gain: 0.33,
        decay: 1.25,
    },
    Mode {
        ratio: 2.49,
        gain: 0.36,
        decay: 1.21,
    },
    Mode {
        ratio: 2.92,
        gain: 0.33,
        decay: 1.17,
    },
    Mode {
        ratio: 3.49,
        gain: 0.53,
        decay: 1.13,
    },
    Mode {
        ratio: 3.83,
        gain: 0.43,
        decay: 1.09,
    },
    Mode {
        ratio: 4.50,
        gain: 0.66,
        decay: 1.06,
    },
    Mode {
        ratio: 4.96,
        gain: 0.56,
        decay: 1.03,
    },
    Mode {
        ratio: 5.66,
        gain: 0.49,
        decay: 1.00,
    },
    Mode {
        ratio: 6.10,
        gain: 0.60,
        decay: 0.97,
    },
    Mode {
        ratio: 6.16,
        gain: 0.58,
        decay: 0.95,
    },
    Mode {
        ratio: 6.75,
        gain: 0.88,
        decay: 0.92,
    },
    Mode {
        ratio: 7.14,
        gain: 0.83,
        decay: 0.90,
    },
    Mode {
        ratio: 7.99,
        gain: 0.75,
        decay: 0.87,
    },
    Mode {
        ratio: 8.43,
        gain: 0.63,
        decay: 0.85,
    },
    Mode {
        ratio: 8.51,
        gain: 0.68,
        decay: 0.83,
    },
    Mode {
        ratio: 9.58,
        gain: 0.77,
        decay: 0.81,
    },
    Mode {
        ratio: 9.74,
        gain: 0.83,
        decay: 0.80,
    },
    Mode {
        ratio: 10.40,
        gain: 0.72,
        decay: 0.78,
    },
    Mode {
        ratio: 11.29,
        gain: 0.88,
        decay: 0.76,
    },
    Mode {
        ratio: 11.19,
        gain: 0.83,
        decay: 0.74,
    },
    Mode {
        ratio: 12.04,
        gain: 0.95,
        decay: 0.73,
    },
    Mode {
        ratio: 12.83,
        gain: 0.72,
        decay: 0.71,
    },
    Mode {
        ratio: 13.70,
        gain: 0.65,
        decay: 0.70,
    },
    Mode {
        ratio: 13.49,
        gain: 0.90,
        decay: 0.69,
    },
    Mode {
        ratio: 13.65,
        gain: 0.80,
        decay: 0.67,
    },
    Mode {
        ratio: 14.00,
        gain: 0.87,
        decay: 0.66,
    },
    Mode {
        ratio: 15.62,
        gain: 0.83,
        decay: 0.65,
    },
    Mode {
        ratio: 16.35,
        gain: 0.73,
        decay: 0.64,
    },
    Mode {
        ratio: 16.61,
        gain: 0.84,
        decay: 0.62,
    },
    Mode {
        ratio: 16.97,
        gain: 0.78,
        decay: 0.61,
    },
    Mode {
        ratio: 17.98,
        gain: 0.98,
        decay: 0.60,
    },
    Mode {
        ratio: 17.89,
        gain: 0.87,
        decay: 0.59,
    },
    Mode {
        ratio: 17.67,
        gain: 0.88,
        decay: 0.58,
    },
    Mode {
        ratio: 19.31,
        gain: 1.00,
        decay: 0.57,
    },
    Mode {
        ratio: 20.21,
        gain: 0.71,
        decay: 0.56,
    },
    Mode {
        ratio: 19.91,
        gain: 0.87,
        decay: 0.56,
    },
    Mode {
        ratio: 19.70,
        gain: 0.78,
        decay: 0.55,
    },
    Mode {
        ratio: 20.54,
        gain: 0.65,
        decay: 0.54,
    },
    Mode {
        ratio: 20.84,
        gain: 0.91,
        decay: 0.53,
    },
    Mode {
        ratio: 21.53,
        gain: 0.70,
        decay: 0.52,
    },
    Mode {
        ratio: 22.67,
        gain: 0.95,
        decay: 0.51,
    },
    Mode {
        ratio: 22.50,
        gain: 0.78,
        decay: 0.51,
    },
    Mode {
        ratio: 24.17,
        gain: 0.95,
        decay: 0.50,
    },
    Mode {
        ratio: 25.40,
        gain: 0.95,
        decay: 0.49,
    },
    Mode {
        ratio: 24.62,
        gain: 0.77,
        decay: 0.49,
    },
];

/// A synth ride: denser and lower than the crash, with a lift in the
/// partials around 8–16× the base — the ping — and long decays. With
/// its lows quiet and only 32 partials the first ride was a triangle
/// with a shimmer; a ride's body is the mids.
pub const RIDE: &[Mode] = &[
    Mode {
        ratio: 1.00,
        gain: 0.49,
        decay: 3.00,
    },
    Mode {
        ratio: 1.33,
        gain: 0.51,
        decay: 2.86,
    },
    Mode {
        ratio: 1.60,
        gain: 0.58,
        decay: 2.73,
    },
    Mode {
        ratio: 1.88,
        gain: 0.60,
        decay: 2.61,
    },
    Mode {
        ratio: 2.32,
        gain: 0.73,
        decay: 2.50,
    },
    Mode {
        ratio: 2.54,
        gain: 0.61,
        decay: 2.40,
    },
    Mode {
        ratio: 2.89,
        gain: 0.83,
        decay: 2.31,
    },
    Mode {
        ratio: 3.45,
        gain: 0.59,
        decay: 2.22,
    },
    Mode {
        ratio: 3.94,
        gain: 1.28,
        decay: 2.14,
    },
    Mode {
        ratio: 4.20,
        gain: 1.10,
        decay: 2.07,
    },
    Mode {
        ratio: 4.37,
        gain: 0.79,
        decay: 2.00,
    },
    Mode {
        ratio: 4.93,
        gain: 0.81,
        decay: 1.94,
    },
    Mode {
        ratio: 5.14,
        gain: 0.91,
        decay: 1.88,
    },
    Mode {
        ratio: 5.44,
        gain: 1.02,
        decay: 1.82,
    },
    Mode {
        ratio: 6.07,
        gain: 1.22,
        decay: 1.76,
    },
    Mode {
        ratio: 6.52,
        gain: 1.11,
        decay: 1.71,
    },
    Mode {
        ratio: 6.91,
        gain: 1.12,
        decay: 1.67,
    },
    Mode {
        ratio: 7.29,
        gain: 0.71,
        decay: 1.62,
    },
    Mode {
        ratio: 8.11,
        gain: 1.00,
        decay: 1.58,
    },
    Mode {
        ratio: 8.42,
        gain: 0.88,
        decay: 1.54,
    },
    Mode {
        ratio: 8.40,
        gain: 0.69,
        decay: 1.50,
    },
    Mode {
        ratio: 8.78,
        gain: 0.63,
        decay: 1.46,
    },
    Mode {
        ratio: 9.64,
        gain: 0.76,
        decay: 1.43,
    },
    Mode {
        ratio: 10.15,
        gain: 0.75,
        decay: 1.40,
    },
    Mode {
        ratio: 10.70,
        gain: 0.94,
        decay: 1.36,
    },
    Mode {
        ratio: 10.13,
        gain: 0.68,
        decay: 1.33,
    },
    Mode {
        ratio: 11.54,
        gain: 0.79,
        decay: 1.30,
    },
    Mode {
        ratio: 12.06,
        gain: 0.76,
        decay: 1.28,
    },
    Mode {
        ratio: 11.43,
        gain: 0.85,
        decay: 1.25,
    },
    Mode {
        ratio: 12.72,
        gain: 0.71,
        decay: 1.22,
    },
    Mode {
        ratio: 12.27,
        gain: 0.73,
        decay: 1.20,
    },
    Mode {
        ratio: 13.85,
        gain: 0.90,
        decay: 1.18,
    },
    Mode {
        ratio: 13.15,
        gain: 0.70,
        decay: 1.15,
    },
    Mode {
        ratio: 13.54,
        gain: 0.62,
        decay: 1.13,
    },
    Mode {
        ratio: 14.98,
        gain: 0.67,
        decay: 1.11,
    },
    Mode {
        ratio: 15.07,
        gain: 0.78,
        decay: 1.09,
    },
    Mode {
        ratio: 14.95,
        gain: 0.89,
        decay: 1.07,
    },
    Mode {
        ratio: 15.28,
        gain: 0.86,
        decay: 1.05,
    },
    Mode {
        ratio: 15.68,
        gain: 0.77,
        decay: 1.03,
    },
    Mode {
        ratio: 16.27,
        gain: 0.71,
        decay: 1.02,
    },
    Mode {
        ratio: 18.01,
        gain: 0.92,
        decay: 1.00,
    },
    Mode {
        ratio: 17.30,
        gain: 0.95,
        decay: 0.98,
    },
    Mode {
        ratio: 17.57,
        gain: 0.76,
        decay: 0.97,
    },
    Mode {
        ratio: 19.19,
        gain: 0.86,
        decay: 0.95,
    },
    Mode {
        ratio: 18.23,
        gain: 1.00,
        decay: 0.94,
    },
    Mode {
        ratio: 18.88,
        gain: 0.70,
        decay: 0.92,
    },
    Mode {
        ratio: 20.43,
        gain: 0.73,
        decay: 0.91,
    },
    Mode {
        ratio: 19.93,
        gain: 0.63,
        decay: 0.90,
    },
];

/// A synth tam-tam: the cymbal's dense inharmonic partials pitched
/// low, with the lows loud (a big thick plate's weight is low), and
/// decays of many seconds. Its bloom — the swell that arrives after
/// the strike — is the pad's bloom with a long delay and spread.
pub const GONG: &[Mode] = &[
    Mode {
        ratio: 0.97,
        gain: 0.86,
        decay: 8.00,
    },
    Mode {
        ratio: 1.48,
        gain: 0.88,
        decay: 7.62,
    },
    Mode {
        ratio: 2.14,
        gain: 0.72,
        decay: 7.27,
    },
    Mode {
        ratio: 2.63,
        gain: 0.95,
        decay: 6.96,
    },
    Mode {
        ratio: 3.38,
        gain: 0.77,
        decay: 6.67,
    },
    Mode {
        ratio: 4.39,
        gain: 0.84,
        decay: 6.40,
    },
    Mode {
        ratio: 5.09,
        gain: 0.84,
        decay: 6.15,
    },
    Mode {
        ratio: 5.77,
        gain: 0.75,
        decay: 5.93,
    },
    Mode {
        ratio: 6.55,
        gain: 0.96,
        decay: 5.71,
    },
    Mode {
        ratio: 7.27,
        gain: 0.92,
        decay: 5.52,
    },
    Mode {
        ratio: 8.20,
        gain: 0.72,
        decay: 5.33,
    },
    Mode {
        ratio: 9.11,
        gain: 0.88,
        decay: 5.16,
    },
    Mode {
        ratio: 9.52,
        gain: 0.71,
        decay: 5.00,
    },
    Mode {
        ratio: 10.94,
        gain: 0.84,
        decay: 4.85,
    },
    Mode {
        ratio: 11.65,
        gain: 0.96,
        decay: 4.71,
    },
    Mode {
        ratio: 12.52,
        gain: 0.98,
        decay: 4.57,
    },
    Mode {
        ratio: 12.99,
        gain: 0.94,
        decay: 4.44,
    },
    Mode {
        ratio: 13.92,
        gain: 0.98,
        decay: 4.32,
    },
    Mode {
        ratio: 15.45,
        gain: 0.73,
        decay: 4.21,
    },
    Mode {
        ratio: 15.20,
        gain: 0.77,
        decay: 4.10,
    },
    Mode {
        ratio: 17.45,
        gain: 0.83,
        decay: 4.00,
    },
    Mode {
        ratio: 17.80,
        gain: 0.79,
        decay: 3.90,
    },
    Mode {
        ratio: 18.50,
        gain: 0.82,
        decay: 3.81,
    },
    Mode {
        ratio: 19.12,
        gain: 0.88,
        decay: 3.72,
    },
    Mode {
        ratio: 20.50,
        gain: 0.97,
        decay: 3.64,
    },
    Mode {
        ratio: 21.65,
        gain: 0.98,
        decay: 3.56,
    },
    Mode {
        ratio: 22.98,
        gain: 1.00,
        decay: 3.48,
    },
    Mode {
        ratio: 23.53,
        gain: 0.75,
        decay: 3.40,
    },
    Mode {
        ratio: 24.95,
        gain: 0.99,
        decay: 3.33,
    },
    Mode {
        ratio: 26.04,
        gain: 0.87,
        decay: 3.27,
    },
    Mode {
        ratio: 26.54,
        gain: 0.46,
        decay: 3.20,
    },
    Mode {
        ratio: 27.84,
        gain: 0.52,
        decay: 3.14,
    },
    Mode {
        ratio: 27.31,
        gain: 0.43,
        decay: 3.08,
    },
    Mode {
        ratio: 29.90,
        gain: 0.60,
        decay: 3.02,
    },
    Mode {
        ratio: 28.62,
        gain: 0.56,
        decay: 2.96,
    },
    Mode {
        ratio: 30.55,
        gain: 0.45,
        decay: 2.91,
    },
    Mode {
        ratio: 31.16,
        gain: 0.56,
        decay: 2.86,
    },
    Mode {
        ratio: 34.02,
        gain: 0.43,
        decay: 2.81,
    },
    Mode {
        ratio: 34.18,
        gain: 0.43,
        decay: 2.76,
    },
    Mode {
        ratio: 35.54,
        gain: 0.48,
        decay: 2.71,
    },
    Mode {
        ratio: 37.14,
        gain: 0.60,
        decay: 2.67,
    },
    Mode {
        ratio: 36.80,
        gain: 0.60,
        decay: 2.62,
    },
    Mode {
        ratio: 37.07,
        gain: 0.43,
        decay: 2.58,
    },
    Mode {
        ratio: 39.18,
        gain: 0.43,
        decay: 2.54,
    },
    Mode {
        ratio: 38.61,
        gain: 0.49,
        decay: 2.50,
    },
    Mode {
        ratio: 41.28,
        gain: 0.45,
        decay: 2.46,
    },
    Mode {
        ratio: 39.93,
        gain: 0.58,
        decay: 2.42,
    },
    Mode {
        ratio: 42.07,
        gain: 0.59,
        decay: 2.39,
    },
];

/// A synth hi-hat: a small, stiff pair, so the partials climb past
/// 10 kHz from a base around 600 Hz with the weight at the top — the
/// brightest thing on the kit. The physical hat (hihat.rs) has the
/// pedal, the choke and the chick as mechanism but not this band: our
/// plate's modes stop near 3 kHz, which is where a hat starts. This is
/// the picture, as CYMBAL is to the modal plate.
pub const HAT: &[Mode] = &[
    Mode {
        ratio: 1.01,
        gain: 0.27,
        decay: 0.90,
    },
    Mode {
        ratio: 1.46,
        gain: 0.33,
        decay: 0.87,
    },
    Mode {
        ratio: 1.96,
        gain: 0.36,
        decay: 0.85,
    },
    Mode {
        ratio: 2.32,
        gain: 0.32,
        decay: 0.83,
    },
    Mode {
        ratio: 3.12,
        gain: 0.38,
        decay: 0.80,
    },
    Mode {
        ratio: 3.69,
        gain: 0.31,
        decay: 0.78,
    },
    Mode {
        ratio: 4.11,
        gain: 0.36,
        decay: 0.76,
    },
    Mode {
        ratio: 4.73,
        gain: 0.45,
        decay: 0.74,
    },
    Mode {
        ratio: 5.05,
        gain: 0.40,
        decay: 0.73,
    },
    Mode {
        ratio: 5.79,
        gain: 0.59,
        decay: 0.71,
    },
    Mode {
        ratio: 6.71,
        gain: 0.43,
        decay: 0.69,
    },
    Mode {
        ratio: 7.37,
        gain: 0.45,
        decay: 0.68,
    },
    Mode {
        ratio: 7.88,
        gain: 0.47,
        decay: 0.66,
    },
    Mode {
        ratio: 8.01,
        gain: 0.72,
        decay: 0.65,
    },
    Mode {
        ratio: 8.81,
        gain: 0.54,
        decay: 0.63,
    },
    Mode {
        ratio: 10.19,
        gain: 0.78,
        decay: 0.62,
    },
    Mode {
        ratio: 10.15,
        gain: 0.85,
        decay: 0.61,
    },
    Mode {
        ratio: 11.07,
        gain: 0.78,
        decay: 0.60,
    },
    Mode {
        ratio: 11.35,
        gain: 0.91,
        decay: 0.58,
    },
    Mode {
        ratio: 12.60,
        gain: 0.95,
        decay: 0.57,
    },
    Mode {
        ratio: 13.55,
        gain: 0.72,
        decay: 0.56,
    },
    Mode {
        ratio: 13.52,
        gain: 0.67,
        decay: 0.55,
    },
    Mode {
        ratio: 13.88,
        gain: 0.63,
        decay: 0.54,
    },
    Mode {
        ratio: 14.77,
        gain: 0.84,
        decay: 0.53,
    },
    Mode {
        ratio: 14.98,
        gain: 0.87,
        decay: 0.52,
    },
    Mode {
        ratio: 16.18,
        gain: 0.72,
        decay: 0.51,
    },
    Mode {
        ratio: 17.69,
        gain: 0.79,
        decay: 0.51,
    },
    Mode {
        ratio: 17.51,
        gain: 0.79,
        decay: 0.50,
    },
    Mode {
        ratio: 18.92,
        gain: 0.62,
        decay: 0.49,
    },
    Mode {
        ratio: 20.16,
        gain: 0.61,
        decay: 0.48,
    },
    Mode {
        ratio: 20.45,
        gain: 0.94,
        decay: 0.47,
    },
    Mode {
        ratio: 19.66,
        gain: 0.92,
        decay: 0.47,
    },
    Mode {
        ratio: 21.09,
        gain: 0.83,
        decay: 0.46,
    },
    Mode {
        ratio: 21.00,
        gain: 0.62,
        decay: 0.45,
    },
    Mode {
        ratio: 22.07,
        gain: 0.98,
        decay: 0.45,
    },
    Mode {
        ratio: 22.81,
        gain: 0.90,
        decay: 0.44,
    },
    Mode {
        ratio: 25.29,
        gain: 0.98,
        decay: 0.43,
    },
    Mode {
        ratio: 24.58,
        gain: 0.74,
        decay: 0.43,
    },
    Mode {
        ratio: 25.76,
        gain: 0.91,
        decay: 0.42,
    },
    Mode {
        ratio: 25.39,
        gain: 0.90,
        decay: 0.41,
    },
    Mode {
        ratio: 27.96,
        gain: 0.94,
        decay: 0.41,
    },
    Mode {
        ratio: 26.60,
        gain: 0.98,
        decay: 0.40,
    },
    Mode {
        ratio: 27.45,
        gain: 0.74,
        decay: 0.40,
    },
    Mode {
        ratio: 29.69,
        gain: 0.97,
        decay: 0.39,
    },
    Mode {
        ratio: 29.62,
        gain: 0.97,
        decay: 0.39,
    },
    Mode {
        ratio: 30.98,
        gain: 0.72,
        decay: 0.38,
    },
    Mode {
        ratio: 31.01,
        gain: 0.67,
        decay: 0.38,
    },
    Mode {
        ratio: 30.97,
        gain: 0.66,
        decay: 0.37,
    },
];

/// A snare head, from a recording of a dry snare (samples/): not an
/// ideal membrane. Tuned tight on a shallow shell, its (1,1) sits at
/// 1.2–1.5× the fundamental as a cluster rather than the membrane's
/// 1.59×, a second group lands around 3.3–4×, and everything is gone
/// in tens of milliseconds — the wires are the note. A snare pad on
/// the MEMBRANE table was a tom with a rattle, and looked it.
pub const SNARE: &[Mode] = &[
    Mode {
        ratio: 1.00,
        gain: 1.00,
        decay: 0.09,
    },
    Mode {
        ratio: 0.73,
        gain: 0.10,
        decay: 0.06,
    },
    Mode {
        ratio: 1.20,
        gain: 0.12,
        decay: 0.07,
    },
    Mode {
        ratio: 1.29,
        gain: 0.28,
        decay: 0.07,
    },
    Mode {
        ratio: 1.46,
        gain: 0.14,
        decay: 0.06,
    },
    Mode {
        ratio: 1.62,
        gain: 0.20,
        decay: 0.06,
    },
    Mode {
        ratio: 2.31,
        gain: 0.10,
        decay: 0.05,
    },
    Mode {
        ratio: 3.28,
        gain: 0.10,
        decay: 0.05,
    },
    Mode {
        ratio: 3.75,
        gain: 0.11,
        decay: 0.05,
    },
    Mode {
        ratio: 4.08,
        gain: 0.12,
        decay: 0.05,
    },
];
