//! Model 3: formant voice — the source-filter model.
//!
//! The glottis makes a buzzy pulse train at the pitch; the vocal tract
//! filters it through resonant peaks (formants). Pitch is the source,
//! *identity* is the filter: move the formants and the same note becomes
//! a different word. A vowel here is just three (frequency, bandwidth,
//! gain) triples — timbre as a handful of numbers again, like the modal
//! drums, except these modes are driven continuously and they move.
//!
//! Syllables fall out of formant motion: start at "ee" and glide to "ah"
//! and you've said "yah"; start at "oo" and you've said "wah".

use crate::util::{Rng, SR, normalize};
use std::f32::consts::{PI, TAU};

/// Three formants: (center frequency Hz, bandwidth Hz, gain).
/// Classic male-voice averages; raise F1/F2 ~15% for a brighter voice.
#[derive(Clone, Copy)]
pub struct Vowel(pub [(f32, f32, f32); 3]);

pub const AH: Vowel = Vowel([
    (730.0, 80.0, 1.0),
    (1090.0, 90.0, 0.50),
    (2440.0, 120.0, 0.20),
]);
pub const EE: Vowel = Vowel([
    (270.0, 60.0, 1.0),
    (2290.0, 90.0, 0.35),
    (3010.0, 120.0, 0.15),
]);
pub const OO: Vowel = Vowel([
    (300.0, 60.0, 1.0),
    (870.0, 80.0, 0.40),
    (2240.0, 120.0, 0.10),
]);
pub const EH: Vowel = Vowel([
    (530.0, 70.0, 1.0),
    (1840.0, 90.0, 0.40),
    (2480.0, 120.0, 0.20),
]);
pub const OH: Vowel = Vowel([
    (570.0, 70.0, 1.0),
    (840.0, 80.0, 0.50),
    (2410.0, 120.0, 0.15),
]);

#[derive(Clone, Copy)]
pub struct Note {
    pub freq: f32,
    pub duration: f32,
    /// Vowel at the onset...
    pub from: Vowel,
    /// ...and the vowel it settles into. Same vowel twice = a plain vowel.
    pub to: Vowel,
    /// Seconds to travel from → to. Short (~0.1) reads as a consonant-ish
    /// onset ("y", "w"); long reads as a diphthong.
    pub glide_time: f32,
    pub attack: f32,
    pub release: f32,
    /// Vibrato depth as a pitch fraction (~0.01). It fades in over the
    /// note, which is most of what makes this read as singing.
    pub vibrato: f32,
    /// Breath: noise mixed into the glottal source. 1.0 is a whisper —
    /// the formants alone still carry the vowel.
    pub breath: f32,
    pub level: f32,
}

impl Default for Note {
    fn default() -> Self {
        Note {
            freq: 175.0,
            duration: 0.6,
            from: AH,
            to: AH,
            glide_time: 0.12,
            attack: 0.04,
            release: 0.12,
            vibrato: 0.012,
            breath: 0.06,
            level: 0.8,
        }
    }
}

/// Rosenberg glottal pulse: the fold opens smoothly, snaps shut, stays
/// closed. The sharp closure is what gives the buzz its high harmonics.
/// (Shared with the streaming mouth.)
pub(crate) fn glottal(p: f32) -> f32 {
    const OPEN: f32 = 0.6;
    const CLOSE: f32 = 0.15;
    if p < OPEN {
        0.5 * (1.0 - (PI * p / OPEN).cos())
    } else if p < OPEN + CLOSE {
        ((p - OPEN) / CLOSE * (PI / 2.0)).cos()
    } else {
        0.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn render(n: &Note, rng: &mut Rng) -> Vec<f32> {
    let len = (n.duration * SR) as usize;
    let mut out = Vec::with_capacity(len);

    let mut phase = 0.0f32;
    // Two-pole resonator state per formant.
    let mut y1 = [0.0f32; 3];
    let mut y2 = [0.0f32; 3];

    for i in 0..len {
        let t = i as f32 / SR;

        // Source: buzz at the pitch, with vibrato fading in.
        let vib_depth = n.vibrato * (t / (n.duration * 0.4)).min(1.0);
        let f = n.freq * (1.0 + vib_depth * (TAU * 5.3 * t).sin());
        phase = (phase + f / SR) % 1.0;
        let src = lerp(glottal(phase), rng.next(), n.breath);

        // Filter: three resonators tracking the moving vowel.
        let glide = (t / n.glide_time).min(1.0);
        let mut sample = 0.0f32;
        for (k, (y1k, y2k)) in y1.iter_mut().zip(y2.iter_mut()).enumerate() {
            let (f0, bw0, g0) = n.from.0[k];
            let (f1, bw1, g1) = n.to.0[k];
            let (fc, bw, gain) = (
                lerp(f0, f1, glide),
                lerp(bw0, bw1, glide),
                lerp(g0, g1, glide),
            );
            let r = (-PI * bw / SR).exp();
            let y = (1.0 - r) * src + 2.0 * r * (TAU * fc / SR).cos() * *y1k - r * r * *y2k;
            *y2k = *y1k;
            *y1k = y;
            sample += gain * y;
        }

        // Envelope: linear attack, linear release.
        let env = (t / n.attack).min(1.0) * ((n.duration - t) / n.release).clamp(0.0, 1.0);
        out.push(sample * env);
    }

    normalize(&mut out, n.level);
    out
}
