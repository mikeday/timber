//! Model 1: Karplus-Strong plucked string.
//!
//! A delay line the length of one period is filled with noise (the pluck),
//! then recirculated through a loss filter. The feedback loop imposes the
//! same periodicity a vibrating string does, so the noise burst decays into
//! a pitched tone. Everything interesting lives in what you put *in* the
//! loop: the loss filter is the string's material, an allpass is its
//! stiffness, the shape of the initial noise is where and how you plucked.

use crate::util::{Rng, SR, fade_out};

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
    // The pitch is set by the *total* delay around the loop, which is
    // rarely a whole number of samples. The integer delay line gets us
    // within a sample; a first-order allpass tuned to the fractional
    // remainder d (coefficient (1-d)/(1+d)) supplies the rest. An allpass
    // rather than interpolation because its magnitude response is flat:
    // it adds delay without adding loss inside the loop. The loss filter
    // (~damping samples at low frequency) and the stiffness allpass
    // ((1-a)/(1+a) samples) contribute delay of their own, so they are
    // subtracted from the budget first.
    let period = SR / p.freq;
    let others = p.damping + (1.0 - p.stiffness) / (1.0 + p.stiffness);
    let target = period - others;
    // Keep d in 0.5..1.5, the allpass's well-behaved range.
    let n = ((target - 0.5).floor().max(2.0)) as usize;
    let d = target - n as f32;
    let tune = (1.0 - d) / (1.0 + d);

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
    let mut ap_x1 = 0.0f32; // stiffness allpass memory
    let mut ap_y1 = 0.0f32;
    let mut tn_x1 = 0.0f32; // tuning allpass memory
    let mut tn_y1 = 0.0f32;
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

        // Fractional tuning: the same allpass structure, aimed at exactly
        // the missing fraction of a sample.
        let tuned = tune * ap + tn_x1 - tune * tn_y1;
        tn_x1 = ap;
        tn_y1 = tuned;

        line[idx] = p.decay * tuned;
        idx = (idx + 1) % n;
    }
    fade_out(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::measured_freq;

    #[test]
    fn pitch_lands_within_two_cents() {
        // 446.16 is deliberately awkward: its period is 98.84 samples,
        // nearly the worst case for integer rounding (~3 cents off before
        // fractional tuning; the highest notes were off by far more).
        for freq in [110.0, 220.0, 446.16, 880.0, 1760.0] {
            let p = Pluck {
                freq,
                duration: 1.0,
                ..Default::default()
            };
            let buf = render(&p, &mut Rng(1));
            let f = measured_freq(&buf[2205..], freq);
            let cents = 1200.0 * (f / freq).log2();
            assert!(
                cents.abs() < 2.0,
                "{freq} Hz came out {f:.2} Hz ({cents:+.1} cents)"
            );
        }
    }
}
