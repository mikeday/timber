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
