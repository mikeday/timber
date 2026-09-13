//! Model 2b: streaming drums — the modal engine as ringing filters.
//!
//! modal.rs renders a strike by summing decaying sines offline. Here the
//! same mode lists live as two-pole resonator filters on the audio
//! thread: a strike is an impulse into the bank, and the knobs act on
//! the *state* — muffle a bell mid-ring, bend a sounding tom. A restrike
//! adds energy to modes already ringing, the way a real head takes a
//! second hit, and a choke is genuine damping (the decay times collapse)
//! rather than a fade on a recording.

use crate::modal::Mode;
use crate::util::{Rng, SR};
use std::f32::consts::TAU;

/// Enough for a synth cymbal's partials; a pad only ticks the modes
/// it has, so the small pads cost nothing extra.
pub const MAX_MODES: usize = 48;

/// Coefficients are refreshed every this many samples — cheap enough to
/// track the pitch glide, rare enough to keep the exp/cos off the
/// per-sample path.
const REFRESH: u32 = 16;

/// Per-sample state decay while choked: ~4 ms to silence.
const CHOKE: f32 = 0.9943;

/// Everything a pad is; plain data the UI ships over on any change.
#[derive(Clone, Copy)]
pub struct PadParams {
    pub modes: &'static [Mode],
    pub freq: f32,
    pub glide: f32,
    pub glide_time: f32,
    pub damp: f32,
    pub noise: f32,
    pub noise_decay: f32,
    /// Color of the noise burst, 0..1: 0 is the bright, flat rattle a
    /// snare or a stick click wants; 1 is a cymbal wash — noise
    /// through a broad resonance that starts high and falls as the
    /// burst decays, the way a crash's roar darkens into its tail.
    pub noise_tone: f32,
    /// Wash bloom, 0..1: a hard hit sends a second, delayed impulse
    /// into the upper partials (growing with strength²), so the roar
    /// is made of the ring's own partials rather than laid over it as
    /// noise — the *outcome* of a cymbal's cascade, without simulating
    /// the cascade.
    pub bloom: f32,
    /// When the bloom arrives after the stick, seconds: ~25 ms is a
    /// cymbal's crack, ~300 ms a tam-tam's swell.
    pub bloom_delay: f32,
    /// Over how long the bloom is delivered, seconds: 0 is one
    /// impulse; longer spreads it as a noise burst of the same energy,
    /// so the partials fill in gradually — the gong's bloom.
    pub bloom_spread: f32,
    pub drive: f32,
    /// Doublet splitting, 0..1: every mode becomes a detuned pair, the
    /// triangle's corner trick generalized. The split is constant in Hz
    /// (2–18 Hz of beat at 1.0, scattered per mode) — proportional splits would push
    /// high modes into 20-150 Hz beating, which reads as buzz.
    pub shimmer: f32,
    pub level: f32,
    pub choke: Option<u8>,
}

#[derive(Clone, Copy, Default)]
struct Resonator {
    y1: f32,
    y2: f32,
    b1: f32,
    b2: f32,
    g: f32,
}

pub struct Pad {
    p: PadParams,
    /// Two banks: primary resonators in slots 0..n, shimmer partners in
    /// MAX_MODES..MAX_MODES+n. With shimmer off and the partner bank
    /// silent the pairs collapse to single full-gain resonators and the
    /// partner bank isn't ticked — the doubled cost is only paid while
    /// actually shimmering.
    res: [Resonator; 2 * MAX_MODES],
    /// Impulse pending injection on the next tick.
    impulse: f32,
    /// The bloom: an impulse weighted toward the high partials, held
    /// back a few ms after the strike.
    bloom_pending: f32,
    bloom_wait: u32,
    /// A spread bloom in progress: samples left and per-sample level.
    bloom_left: u32,
    bloom_amp: f32,
    impulse_hi: f32,
    glide_env: f32,
    glide_step: f32,
    rattle: f32,
    rattle_step: f32,
    rattle_prev: f32,
    /// The burst's level at the strike (for the wash's brightness
    /// envelope) and the wash resonator's state.
    rattle0: f32,
    wash_y1: f32,
    wash_y2: f32,
    choking: bool,
    refresh: u32,
    rolling: bool,
    /// Samples until the next roll restrike.
    roll_t: f32,
    // Smoothed so slider sweeps ride ringing sound without zipper.
    damp_s: f32,
    freq_s: f32,
    shimmer_s: f32,
    collapsed: bool,
}

fn smooth(cur: &mut f32, target: f32) {
    *cur += 0.002 * (target - *cur);
}

impl Pad {
    fn new(p: PadParams) -> Self {
        Pad {
            p,
            res: [Resonator::default(); 2 * MAX_MODES],
            impulse: 0.0,
            bloom_pending: 0.0,
            bloom_wait: 0,
            bloom_left: 0,
            bloom_amp: 0.0,
            impulse_hi: 0.0,
            glide_env: 0.0,
            glide_step: 1.0,
            rattle: 0.0,
            rattle_step: 1.0,
            rattle_prev: 0.0,
            rattle0: 1.0,
            wash_y1: 0.0,
            wash_y2: 0.0,
            choking: false,
            refresh: 0,
            rolling: false,
            roll_t: 0.0,
            damp_s: p.damp,
            freq_s: p.freq,
            shimmer_s: p.shimmer,
            collapsed: p.shimmer < 1e-3,
        }
    }

    fn strike(&mut self, strength: f32) {
        // Normalize the impulse by the bank's total gain so a pad's peak
        // tracks `level` regardless of how many modes it carries — but
        // count only modes that will actually sound: a high-tuned pad's
        // Nyquist-muted modes must not dilute the hit.
        let sum: f32 = self
            .p
            .modes
            .iter()
            .take(MAX_MODES)
            .filter(|m| self.freq_s * m.ratio < 0.45 * SR)
            .map(|m| m.gain)
            .sum();
        if sum > 0.0 {
            self.impulse += strength / sum;
            if self.p.bloom > 0.0 {
                self.bloom_pending += self.p.bloom * strength * strength * 2.0 / sum;
                self.bloom_wait = (self.p.bloom_delay.clamp(0.0, 2.0) * SR) as u32;
            }
        }
        self.glide_env = 1.0;
        self.glide_step = (-1.0 / (self.p.glide_time.max(0.005) * SR)).exp();
        self.rattle = self.p.noise * 0.5;
        self.rattle0 = self.rattle.max(1e-9);
        self.rattle_step = (-1.0 / (self.p.noise_decay.max(0.002) * SR)).exp();
        self.choking = false;
        self.refresh = 0;
    }

    fn refresh_coeffs(&mut self) {
        let fm = 1.0 + self.p.glide * self.glide_env;
        // Shimmer: constant-Hz split (proportional splits push high
        // modes into 20–150 Hz beating, which reads as buzz), varied
        // per mode — identical splits made every pair beat in step,
        // a coherent tremolo that on a 48-partial cymbal was a slow
        // wah over the whole sound. Real doublets split unevenly.
        let split0 = self.shimmer_s * 4.0;
        self.collapsed = self.shimmer_s < 1e-3
            && self.res[MAX_MODES..]
                .iter()
                .all(|r| r.y1.abs() < 1e-6 && r.y2.abs() < 1e-6);
        for (k, m) in self.p.modes.iter().take(MAX_MODES).enumerate() {
            let f = self.freq_s * m.ratio * fm;
            // 0.3..2.2 of the nominal split, scattered by mode index.
            // Beats below ~4 Hz read as a slow wah on a long-ringing
            // partial, above it as shimmer: long sustains want a
            // higher shimmer setting, not a lower one.
            let split = split0 * (0.3 + 1.9 * ((k * 37 + 11) % 23) as f32 / 22.0);
            let pole = (-1.0 / ((m.decay * self.damp_s).max(0.002) * SR)).exp();
            let primary_gain = if self.collapsed { m.gain } else { 0.5 * m.gain };
            let halves = [
                (k, f - split, primary_gain),
                (MAX_MODES + k, f + split, 0.5 * m.gain),
            ];
            for (slot, f, gain) in halves {
                let r = &mut self.res[slot];
                if f >= 0.45 * SR || f <= 0.0 {
                    (r.b1, r.b2, r.g) = (0.0, 0.0, 0.0);
                    continue;
                }
                let th = TAU * f / SR;
                r.b1 = 2.0 * pole * th.cos();
                r.b2 = pole * pole;
                // sin(θ) input normalization, without which a low mode
                // rings up 1/sin(θ) ≈ 100× louder than a high one from
                // the same hit.
                r.g = gain * th.sin();
            }
        }
        // A finished choke clears the bank so denormal-tiny states don't
        // linger.
        if self.choking
            && self.rattle < 1e-6
            && self.bloom_pending <= 0.0
            && self.bloom_left == 0
            && self
                .res
                .iter()
                .all(|r| r.y1.abs() < 1e-6 && r.y2.abs() < 1e-6)
        {
            self.res = [Resonator::default(); 2 * MAX_MODES];
            self.choking = false;
        }
    }

    fn tick(&mut self, rng: &mut Rng) -> f32 {
        smooth(&mut self.damp_s, self.p.damp);
        smooth(&mut self.freq_s, self.p.freq);
        smooth(&mut self.shimmer_s, self.p.shimmer);
        self.glide_env *= self.glide_step;
        if self.refresh == 0 {
            self.refresh = REFRESH;
            self.refresh_coeffs();
        }
        self.refresh -= 1;

        let x = self.impulse;
        self.impulse = 0.0;
        if self.bloom_pending > 0.0 {
            if self.bloom_wait == 0 {
                let spread = (self.p.bloom_spread.clamp(0.0, 2.0) * SR) as u32;
                if spread == 0 {
                    self.impulse_hi = self.bloom_pending;
                } else {
                    // Same energy as the single impulse, as noise over
                    // the spread: random phases add in power, so the
                    // per-sample level is the impulse over √N.
                    self.bloom_left = spread;
                    self.bloom_amp = self.bloom_pending / (spread as f32).sqrt();
                }
                self.bloom_pending = 0.0;
            } else {
                self.bloom_wait -= 1;
            }
        }
        let mut x_hi = self.impulse_hi;
        self.impulse_hi = 0.0;
        if self.bloom_left > 0 {
            x_hi += self.bloom_amp * rng.next();
            self.bloom_left -= 1;
        }
        let n = self.p.modes.len().min(MAX_MODES);
        let choking = self.choking;
        // Bloom weight rises with the mode's place in the list — the
        // tables put the high partials last — so the second impulse
        // lands mostly in the top of the spectrum.
        let inv_n = 1.0 / n.max(1) as f32;
        let run = |k: usize, r: &mut Resonator| {
            let w = (k as f32 * inv_n).powi(2);
            let y = r.b1 * r.y1 - r.b2 * r.y2 + r.g * (x + x_hi * w);
            r.y2 = r.y1;
            r.y1 = y;
            if choking {
                r.y1 *= CHOKE;
                r.y2 *= CHOKE;
            }
            y
        };
        let mut sum = 0.0;
        for (k, r) in self.res[..n].iter_mut().enumerate() {
            sum += run(k, r);
        }
        if !self.collapsed {
            for (k, r) in self.res[MAX_MODES..MAX_MODES + n].iter_mut().enumerate() {
                sum += run(k, r);
            }
        }

        if self.rattle > 1e-6 {
            let n = rng.next();
            let white = (n - self.rattle_prev) * self.rattle;
            self.rattle_prev = n;
            let tone = self.p.noise_tone.clamp(0.0, 1.0);
            if tone > 0.0 {
                // Broad resonance sliding from ~7 kHz down to ~1.5 kHz
                // as the burst decays.
                let frac = (self.rattle / self.rattle0).min(1.0);
                let fc = 1500.0 + 5500.0 * frac.sqrt();
                let th = std::f32::consts::TAU * fc / SR;
                let r = 0.85;
                let y = n * self.rattle + 2.0 * r * th.cos() * self.wash_y1 - r * r * self.wash_y2;
                self.wash_y2 = self.wash_y1;
                self.wash_y1 = y;
                sum += white * (1.0 - tone) + y * 0.2 * tone;
            } else {
                sum += white;
            }
            self.rattle *= if self.choking {
                CHOKE
            } else {
                self.rattle_step
            };
        }

        if self.p.drive > 0.0 {
            sum = (self.p.drive * sum).tanh() / self.p.drive.tanh();
        }
        sum * self.p.level
    }
}

/// The kit: one streaming pad per drum, mixed to a single output.
pub struct Kit {
    pads: Vec<Pad>,
}

impl Kit {
    pub fn new(params: Vec<PadParams>) -> Self {
        Kit {
            pads: params.into_iter().map(Pad::new).collect(),
        }
    }

    /// Replace a pad's parameters; its ringing state carries on under
    /// the new settings — that's the point.
    pub fn set_params(&mut self, i: usize, p: PadParams) {
        if let Some(pad) = self.pads.get_mut(i) {
            // Clear resonators beyond the new mode list: a shorter list
            // would otherwise freeze ringing state mid-vibration, to be
            // re-driven as a ghost burst by a later longer list (and to
            // jam the choke-finished cleanup, which scans every slot).
            let n = p.modes.len().min(MAX_MODES);
            for k in n..MAX_MODES {
                pad.res[k] = Resonator::default();
                pad.res[MAX_MODES + k] = Resonator::default();
            }
            pad.p = p;
            pad.refresh = 0;
        }
    }

    pub fn strike(&mut self, i: usize) {
        self.strike_with(i, 1.0);
    }

    fn strike_with(&mut self, i: usize, strength: f32) {
        if i >= self.pads.len() {
            return;
        }
        if let Some(group) = self.pads[i].p.choke {
            for (j, pad) in self.pads.iter_mut().enumerate() {
                if j != i && pad.p.choke == Some(group) {
                    pad.choking = true;
                }
            }
        }
        self.pads[i].strike(strength);
    }

    /// Hold-to-roll: while on, the pad restrikes itself with humanized
    /// timing and strength. The initial press's strike is separate, so
    /// the timer starts a full interval out.
    pub fn set_roll(&mut self, i: usize, on: bool) {
        if let Some(pad) = self.pads.get_mut(i) {
            pad.rolling = on;
            if on {
                pad.roll_t = 0.06 * SR;
            }
        }
    }

    pub fn tick(&mut self, rng: &mut Rng) -> f32 {
        // Roll restrikes fire at Kit level so they get the same
        // choke-group treatment as manual strikes — a Pad-local roll
        // bypassed the scan and even un-choked itself.
        for i in 0..self.pads.len() {
            if self.pads[i].rolling {
                self.pads[i].roll_t -= 1.0;
                if self.pads[i].roll_t <= 0.0 {
                    self.strike_with(i, 0.6 + 0.35 * rng.next().abs());
                    self.pads[i].roll_t = (0.055 + 0.02 * rng.next().abs()) * SR;
                }
            }
        }
        self.pads.iter_mut().map(|p| p.tick(rng)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modal;
    use crate::util::measured_freq;

    fn tom() -> PadParams {
        PadParams {
            modes: modal::MEMBRANE,
            freq: 110.0,
            glide: 0.25,
            glide_time: 0.12,
            damp: 1.0,
            noise: 0.0,
            noise_decay: 0.1,
            noise_tone: 0.0,
            bloom: 0.0,
            bloom_delay: 0.025,
            bloom_spread: 0.0,
            drive: 0.0,
            shimmer: 0.0,
            level: 0.8,
            choke: None,
        }
    }

    fn bell() -> PadParams {
        PadParams {
            modes: modal::BELL,
            freq: 440.0,
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
            level: 0.8,
            choke: Some(0),
        }
    }

    fn hat() -> PadParams {
        PadParams {
            modes: &[],
            freq: 110.0,
            glide: 0.0,
            glide_time: 0.1,
            damp: 1.0,
            noise: 1.0,
            noise_decay: 0.025,
            noise_tone: 0.0,
            bloom: 0.0,
            bloom_delay: 0.025,
            bloom_spread: 0.0,
            drive: 0.0,
            shimmer: 0.0,
            level: 0.4,
            choke: Some(0),
        }
    }

    fn run(kit: &mut Kit, rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| kit.tick(rng)).collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    #[test]
    fn roll_chokes_siblings() {
        // Roll restrikes must honor choke groups like manual strikes
        // (regression: Pad-local rolls bypassed the Kit-level scan).
        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell(), hat()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 8820);
        kit.set_roll(1, true); // rolling hat shares group 0 with bell
        run(&mut kit, &mut rng, 17640);
        kit.set_roll(1, false);
        run(&mut kit, &mut rng, 8820);
        let after = run(&mut kit, &mut rng, 8820);

        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell(), hat()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 35280);
        let control = run(&mut kit, &mut rng, 8820);

        assert!(
            rms(&after) < rms(&control) * 0.1,
            "roll failed to choke sibling: {} vs control {}",
            rms(&after),
            rms(&control)
        );
    }

    #[test]
    fn mode_swap_mid_ring_leaves_no_ghost() {
        // Swapping to a shorter mode list mid-ring must not freeze
        // resonator state that a later longer list re-drives as a
        // ghost burst (regression: tick only updated the first 2n
        // slots).
        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 8820);
        kit.set_params(
            0,
            PadParams {
                modes: modal::MEMBRANE_CENTER,
                ..bell()
            },
        );
        run(&mut kit, &mut rng, 2 * 44100); // short list rings out fully
        kit.set_params(0, bell());
        let after = run(&mut kit, &mut rng, 22050);
        assert!(
            rms(&after) < 0.01,
            "ghost ring after mode-list swap: {}",
            rms(&after)
        );
    }

    #[test]
    fn roll_holds_level_then_stops() {
        let mut kit = Kit::new(vec![tom()]);
        let mut rng = Rng(3);
        kit.strike(0);
        kit.set_roll(0, true);
        let out = run(&mut kit, &mut rng, 44100);
        let rolling = rms(&out[33075..]);
        assert!(
            rolling > rms(&out[..11025]) * 0.5,
            "roll failed to sustain: {rolling}"
        );
        kit.set_roll(0, false);
        let tail = run(&mut kit, &mut rng, 44100);
        assert!(rms(&tail[22050..]) < rolling * 0.3, "roll failed to stop");
    }

    #[test]
    fn shimmer_makes_the_ring_beat() {
        // One long mode: with shimmer, its envelope must pulse (the
        // split pair beating); without, it just decays smoothly.
        const ONE: &[Mode] = &[Mode {
            ratio: 1.0,
            gain: 1.0,
            decay: 3.0,
        }];
        let contrast = |shimmer: f32| {
            let mut kit = Kit::new(vec![PadParams {
                modes: ONE,
                freq: 500.0,
                shimmer,
                ..tom()
            }]);
            let mut rng = Rng(3);
            kit.strike(0);
            let out = run(&mut kit, &mut rng, 44100);
            let (mut lo, mut hi) = (f32::MAX, 0.0f32);
            for w in out[4410..].chunks(882) {
                let r = rms(w);
                lo = lo.min(r);
                hi = hi.max(r);
            }
            hi / lo.max(1e-9)
        };
        let with = contrast(1.0);
        let without = contrast(0.0);
        assert!(
            with > without * 2.0,
            "no beat from shimmer: {with:.2} vs {without:.2}"
        );
    }

    #[test]
    fn strike_rings_at_pitch_and_decays() {
        let mut kit = Kit::new(vec![tom()]);
        let mut rng = Rng(3);
        kit.strike(0);
        let out = run(&mut kit, &mut rng, 44100);
        assert!(out.iter().all(|s| s.is_finite()));
        // After the glide settles, the fundamental should sit at freq.
        let f = measured_freq(&out[13230..], 110.0);
        assert!((f / 110.0 - 1.0).abs() < 0.03, "tom at {f:.1} Hz");
        let early = rms(&out[0..8820]);
        let late = rms(&out[35280..]);
        assert!(early > 0.02, "strike too quiet: {early}");
        assert!(late < early * 0.5, "no decay: {early} -> {late}");
    }

    #[test]
    fn muffle_strangles_a_ringing_bell() {
        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 22050);
        // Pillow on at t=0.5s.
        kit.set_params(
            0,
            PadParams {
                damp: 0.05,
                ..bell()
            },
        );
        let muffled = run(&mut kit, &mut rng, 22050);

        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 22050);
        let open = run(&mut kit, &mut rng, 22050);

        assert!(
            rms(&muffled[11025..]) < rms(&open[11025..]) * 0.2,
            "muffle did not strangle the ring: {} vs {}",
            rms(&muffled[11025..]),
            rms(&open[11025..])
        );
    }

    #[test]
    fn hat_chokes_the_ringing_bell() {
        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell(), hat()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 8820);
        kit.strike(1); // same choke group: bell must die
        run(&mut kit, &mut rng, 8820);
        let after = run(&mut kit, &mut rng, 8820);

        let mut rng = Rng(3);
        let mut kit = Kit::new(vec![bell(), hat()]);
        kit.strike(0);
        run(&mut kit, &mut rng, 17640);
        let control = run(&mut kit, &mut rng, 8820);

        assert!(
            rms(&after) < rms(&control) * 0.1,
            "choke failed: {} vs control {}",
            rms(&after),
            rms(&control)
        );
    }
}
