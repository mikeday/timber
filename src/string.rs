//! Model 1: Karplus-Strong plucked string.
//!
//! A delay line the length of one period is filled with noise (the pluck),
//! then recirculated through a loss filter. The feedback loop imposes the
//! same periodicity a vibrating string does, so the noise burst decays into
//! a pitched tone. Everything interesting lives in what you put *in* the
//! loop: the loss filter is the string's material, an allpass is its
//! stiffness, the shape of the initial noise is where and how you plucked.

use crate::util::{Rng, SR};

/// One plucked note. Mutate these fields — that's the whole point.
#[derive(Clone, Copy)]
pub struct Pluck {
    pub freq: f32,
    /// Seconds of tail to render.
    pub duration: f32,
    /// Loop gain per period, just below 1.0. 0.999 rings; 0.98 is a thud.
    pub decay: f32,
    /// Loss-filter blend, 0.0 bright (steel) .. 1.0 dark (nylon, felt).
    /// 0.5 is the classic Karplus-Strong two-point average.
    pub damping: f32,
    /// Where along the string you plucked, 0.0..0.5. Near the bridge
    /// (small values) is twangy; toward the middle is round and hollow.
    /// 0.0 disables the comb entirely.
    pub pluck_pos: f32,
    /// Allpass coefficient, 0.0..~0.6. Stretches upper partials sharp,
    /// the way a stiff piano string does. Also flattens the pitch a bit.
    pub stiffness: f32,
    /// Amplitude, 0.0..1.0.
    pub level: f32,
}

impl Default for Pluck {
    fn default() -> Self {
        Pluck {
            freq: 220.0,
            duration: 3.0,
            decay: 0.996,
            damping: 0.5,
            pluck_pos: 0.2,
            stiffness: 0.0,
            level: 0.8,
        }
    }
}

pub fn render(p: &Pluck, rng: &mut Rng) -> Vec<f32> {
    // Delay length sets the pitch. Rounding costs a few cents at low
    // frequencies; a fractional-delay allpass fixes that, later.
    let n = (SR / p.freq).round().max(2.0) as usize;

    // The pluck: one period of noise.
    let mut line: Vec<f32> = (0..n).map(|_| rng.next()).collect();

    // Pluck position: subtracting a delayed copy of the excitation puts a
    // node at that point on the string, killing the harmonics that would
    // need to move there.
    if p.pluck_pos > 0.0 {
        let d = ((p.pluck_pos * n as f32) as usize).clamp(1, n - 1);
        let burst = line.clone();
        for i in 0..n {
            line[i] -= burst[(i + n - d) % n];
        }
    }

    let len = (p.duration * SR) as usize;
    let mut out = Vec::with_capacity(len);
    let mut prev = 0.0f32; // loss-filter memory
    let mut ap_x1 = 0.0f32; // allpass memory
    let mut ap_y1 = 0.0f32;
    let mut idx = 0usize;

    for _ in 0..len {
        let cur = line[idx];
        out.push(cur * p.level);

        // Loss filter: a damped average of neighboring samples. This is
        // the energy the string loses each period — the material.
        let lowpassed = (1.0 - p.damping) * cur + p.damping * prev;
        prev = cur;

        // Stiffness: a first-order allpass delays low frequencies more
        // than highs, stretching the partials inharmonic.
        let a = p.stiffness;
        let ap = a * lowpassed + ap_x1 - a * ap_y1;
        ap_x1 = lowpassed;
        ap_y1 = ap;

        line[idx] = p.decay * ap;
        idx = (idx + 1) % n;
    }
    out
}
