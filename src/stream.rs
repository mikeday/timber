//! Model 4: streaming strings — persistent digital-waveguide strings
//! owned by the audio thread and steered live by the UI. Where string.rs
//! renders a finished note, this keeps the string *existing*: pluck it,
//! bow it, change its material while it rings.
//!
//! Each string is two delay lines meeting at the bow point: the
//! bridge-side and nut-side halves of the string, with an inverting
//! reflection at each end (losses live in the bridge reflection). A bow
//! must sit *inside* the string like this because it launches waves in
//! both directions; feeding a single loop instead turns the bow point
//! into a clamp at high grip, which strangles the fundamental and
//! squeals. (The two inverting reflections also cancel DC per round
//! trip, so no DC-blocker — and none of its phase-lead detuning.)
//!
//! The bow itself is stick-slip friction. While bow and string move
//! together (sticking), friction is strong and drags the string along;
//! when the string breaks away (slipping), friction collapses until the
//! bow catches it again. That catch-and-release, repeating at the
//! string's own round-trip period, is the Helmholtz motion — a bowed
//! tone is a self-oscillation powered by the difference between bow
//! speed and string speed. Which is why a bow needs continuous control:
//! the note only exists while energy flows.

use crate::util::{AtomicF32, Rng, SR};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

/// Longest supported period: A0 at 27.5 Hz.
const MAX_DELAY: usize = 1608;

/// Where along the string the bow sits, as a fraction of its length
/// from the bridge. Close to the bridge (STK's 0.127) needs precise,
/// heavy pressure to speak — skilled-player territory; further out,
/// toward the fingerboard, the fundamental speaks easily across a wide
/// pressure range, which is what a trackpad bow wants. Kept away from
/// simple fractions: at 0.3 ≈ 1/3 the bow sits on the 3rd harmonic's
/// node, where friction can neither feed nor damp it, and it rings
/// unsupervised over slow strokes.
const BETA: f32 = 0.22;

/// Parameter smoothing coefficient (~11 ms). The UI writes at frame
/// rate; stepping a loop parameter 60 times a second would zipper.
const SMOOTH: f32 = 0.002;

/// The control surface the UI writes and the audio thread reads.
pub struct Ctl {
    pub decay: AtomicF32,
    pub damping: AtomicF32,
    pub stiffness: AtomicF32,
    pub pluck_pos: AtomicF32,
    pub level: AtomicF32,
    /// Sympathetic coupling: the fraction of each string's bridge wave
    /// the shared bridge hands to the other strings — PER ROUND TRIP,
    /// which is milliseconds, so honest values are tiny (~0.001-0.01).
    /// The donated fraction is subtracted from the string's own
    /// reflection (passivity demands it), so sympathy is always paid
    /// for out of sustain; audible blooming comes from the receiving
    /// strings' high Q integrating the trickle, not from a big c.
    pub couple: AtomicF32,
    pub bow_on: AtomicBool,
    /// Signed bow velocity, roughly -1..1: the hand. Reversing sign is a
    /// bow change, complete with re-attack scratch.
    pub bow_speed: AtomicF32,
    /// How hard the bow presses, 0..1. Light = airy, whistly harmonics;
    /// heavy = broad grip that locks the fundamental.
    pub bow_pressure: AtomicF32,
    /// Which string the bow currently touches.
    pub bow_string: AtomicUsize,
    /// Hurdy-gurdy mode: the bow touches every string at once, each
    /// locking to its own pitch — a continuously excited drone, which
    /// is how real drone instruments actually drone (sympathy alone
    /// only charges shared partials).
    pub drone: AtomicBool,
}

impl Ctl {
    pub fn new() -> Self {
        Ctl {
            decay: AtomicF32::new(0.996),
            damping: AtomicF32::new(0.5),
            stiffness: AtomicF32::new(0.0),
            pluck_pos: AtomicF32::new(0.2),
            level: AtomicF32::new(0.8),
            couple: AtomicF32::new(0.003),
            bow_on: AtomicBool::new(false),
            bow_speed: AtomicF32::new(0.0),
            bow_pressure: AtomicF32::new(0.5),
            bow_string: AtomicUsize::new(0),
            drone: AtomicBool::new(false),
        }
    }
}

impl Default for Ctl {
    fn default() -> Self {
        Self::new()
    }
}

/// A plain snapshot of Ctl, read once per audio callback.
#[derive(Clone, Copy)]
pub struct Params {
    pub decay: f32,
    pub damping: f32,
    pub stiffness: f32,
    pub level: f32,
    pub couple: f32,
    pub bow_on: bool,
    pub bow_speed: f32,
    pub bow_pressure: f32,
    pub bow_string: usize,
    pub drone: bool,
}

impl Params {
    pub fn read(ctl: &Ctl) -> Self {
        Params {
            decay: ctl.decay.get(),
            damping: ctl.damping.get(),
            stiffness: ctl.stiffness.get(),
            level: ctl.level.get(),
            couple: ctl.couple.get().clamp(0.0, 0.5),
            bow_on: ctl.bow_on.load(Relaxed),
            bow_speed: ctl.bow_speed.get(),
            bow_pressure: ctl.bow_pressure.get(),
            bow_string: ctl.bow_string.load(Relaxed),
            drone: ctl.drone.load(Relaxed),
        }
    }
}

/// STK-style friction curve: ~1 near zero relative velocity (sticking),
/// collapsing steeply once slipping.
fn friction(dv: f32) -> f32 {
    let d = dv.abs() + 0.75;
    (1.0 / (d * d * d * d)).min(1.0)
}

/// One string: two delay lines (bridge side, nut side) joined at the
/// bow point.
pub struct Voice {
    freq: f32,
    bridge: Vec<f32>,
    bridge_w: usize,
    neck: Vec<f32>,
    neck_w: usize,
    prev: f32,  // loss-filter memory (bridge reflection)
    ap_x1: f32, // stiffness allpass memory
    ap_y1: f32,
    // Smoothed local copies of the shared parameters.
    damping: f32,
    decay: f32,
    stiffness: f32,
    couple: f32,
    bow_speed: f32,
    bow_pressure: f32,
}

fn smooth(cur: &mut f32, target: f32) {
    *cur += SMOOTH * (target - *cur);
}

/// Fractional-delay read: the sample written `delay` samples before the
/// next write at `widx`, linearly interpolated.
fn read(line: &[f32], widx: usize, delay: f32) -> f32 {
    let i = delay as usize;
    let frac = delay - i as f32;
    let k0 = (widx + MAX_DELAY - i) % MAX_DELAY;
    let k1 = (k0 + MAX_DELAY - 1) % MAX_DELAY;
    (1.0 - frac) * line[k0] + frac * line[k1]
}

impl Voice {
    fn new(freq: f32) -> Self {
        Voice {
            freq,
            bridge: vec![0.0; MAX_DELAY],
            bridge_w: 0,
            neck: vec![0.0; MAX_DELAY],
            neck_w: 0,
            prev: 0.0,
            ap_x1: 0.0,
            ap_y1: 0.0,
            damping: 0.5,
            decay: 0.996,
            stiffness: 0.0,
            couple: 0.0,
            bow_speed: 0.0,
            bow_pressure: 0.0,
        }
    }

    /// Split of the total loop delay between the two lines. The total is
    /// the period minus what the bridge filters spend, as in string.rs;
    /// the fraction is handled by interpolated reads, which stay
    /// well-behaved while parameters (and so the delay) move.
    fn delays(&self) -> (f32, f32) {
        let a = self.stiffness;
        let total = (SR / self.freq - self.damping - (1.0 - a) / (1.0 + a)).max(4.0);
        let bl = (total * BETA).max(2.0);
        (bl, (total - bl).max(2.0))
    }

    /// Add a noise burst into the nut-side line — the same excitation
    /// string.rs starts from, injected mid-flight.
    fn pluck(&mut self, pos: f32, rng: &mut Rng) {
        let (_, nl) = self.delays();
        let n = (nl as usize).clamp(2, MAX_DELAY);
        let mut burst: Vec<f32> = (0..n).map(|_| rng.next()).collect();
        if pos > 0.0 {
            let d = ((pos * n as f32) as usize).clamp(1, n - 1);
            let orig = burst.clone();
            for i in 0..n {
                burst[i] -= orig[(i + n - d) % n];
            }
        }
        let start = (self.neck_w + MAX_DELAY - n) % MAX_DELAY;
        for (i, b) in burst.iter().enumerate() {
            self.neck[(start + i) % MAX_DELAY] += b;
        }
    }

    /// `cross` is the average of the other strings' bridge waves from
    /// the previous sample; the return is (audible output, own bridge
    /// wave for the pool).
    fn tick(&mut self, p: &Params, bowed: bool, cross: f32) -> (f32, f32) {
        smooth(&mut self.damping, p.damping);
        smooth(&mut self.decay, p.decay);
        smooth(&mut self.stiffness, p.stiffness);
        smooth(&mut self.couple, p.couple);
        let on = bowed && p.bow_on;
        smooth(&mut self.bow_speed, if on { p.bow_speed } else { 0.0 });
        smooth(
            &mut self.bow_pressure,
            if on { p.bow_pressure } else { 0.0 },
        );

        let (bl, nl) = self.delays();
        let b_out = read(&self.bridge, self.bridge_w, bl);
        let n_out = read(&self.neck, self.neck_w, nl);

        // Bridge reflection: inverting, and it carries the string's
        // losses — material lowpass, stiffness allpass, decay.
        let a = self.stiffness;
        let lp = (1.0 - self.damping) * b_out + self.damping * self.prev;
        self.prev = b_out;
        let ap = a * lp + self.ap_x1 - a * self.ap_y1;
        self.ap_x1 = lp;
        self.ap_y1 = ap;
        let f = self.decay * ap;

        // Sympathetic coupling: the bridge is shared, and a passive
        // bridge redistributes rather than creates — each string gets
        // (1-c) of its own wave back and c of the others' pool. Naively
        // *adding* neighbors' signal instead builds a gain loop between
        // strings that share a resonance and self-oscillates; swapping
        // conserves energy at any coupling strength.
        let c = self.couple;
        let br = -((1.0 - c) * f + c * cross);

        // Nut reflection: inverting, lossless.
        let nr = -n_out;

        // The string's velocity under the bow is what both arriving
        // waves say it is; the bow's friction force is launched equally
        // into both halves.
        let vs = br + nr;
        let mut force = 0.0;
        if self.bow_pressure > 1e-4 {
            let bow = self.bow_speed * 0.7;
            let dv = bow - vs;
            // Pressure widens the sticking region: a light bow grips
            // only near zero relative velocity, a heavy one over a wide
            // range. The curve is normalized by the bow speed so the
            // stick-slip geometry scales with the stroke, as the
            // Helmholtz solution itself does — with a fixed knee, fast
            // bowing reads every excursion as slip and loudness
            // plateaus instead of tracking the bow.
            let slope = 1.2 - 0.9 * self.bow_pressure;
            // Half per line: the string's velocity is the SUM of the two
            // arriving waves, so full-stick correction must split across
            // them — vs + 2·(dv/2) = bow, exact stick. Injecting dv into
            // both lines overshoots to 2·bow − vs, a sample-rate
            // flip-flop that pumps spurious energy (loudest, buzziest
            // at slow bowing where sticking dominates).
            force = 0.5 * dv * friction(slope * dv / (0.3 + bow.abs()));
        }

        // Safety: keep runaway states finite, but softly — a hard clamp
        // is a hard clipper.
        let soft = |x: f32| 3.0 * (x / 3.0).tanh();
        self.neck[self.neck_w] = soft(br + force);
        self.neck_w = (self.neck_w + 1) % MAX_DELAY;
        self.bridge[self.bridge_w] = soft(nr + force);
        self.bridge_w = (self.bridge_w + 1) % MAX_DELAY;
        // What you hear is the wave arriving at the bridge — that's what
        // drives a body. The velocity under the bow itself is a stick
        // plateau with slip spikes: nasal, buzzy, and not what a real
        // instrument radiates.
        (br, f)
    }
}

/// A fixed set of open strings, one per pitch — an eight-string zither.
/// The bow touches one of them; plucks land on any.
pub struct Bank {
    voices: Vec<Voice>,
    /// Each string's bridge wave from the previous sample — the pool
    /// the shared bridge redistributes (one-sample delay keeps the
    /// exchange causal and adds no energy).
    pool: Vec<f32>,
}

impl Bank {
    pub fn new(freqs: &[f32]) -> Self {
        Bank {
            voices: freqs.iter().map(|&f| Voice::new(f)).collect(),
            pool: vec![0.0; freqs.len()],
        }
    }

    pub fn pluck(&mut self, string: usize, pos: f32, rng: &mut Rng) {
        if let Some(v) = self.voices.get_mut(string) {
            v.pluck(pos, rng);
        }
    }

    pub fn tick(&mut self, p: &Params) -> f32 {
        let total: f32 = self.pool.iter().sum();
        let others = (self.pool.len().max(2) - 1) as f32;
        let mut sum = 0.0;
        for (i, v) in self.voices.iter_mut().enumerate() {
            let cross = (total - self.pool[i]) / others;
            let (out, f) = v.tick(p, p.drone || i == p.bow_string, cross);
            self.pool[i] = f;
            sum += out;
        }
        sum * p.level
    }

    /// Copy the nut-side line's live window into `out` — the string's
    /// current motion, for drawing.
    pub fn shape(&self, string: usize, out: &mut [f32]) {
        let Some(v) = self.voices.get(string) else {
            return;
        };
        let (_, nl) = v.delays();
        let n = (nl as usize).clamp(2, MAX_DELAY);
        let m = out.len().max(2);
        for (j, o) in out.iter_mut().enumerate() {
            let k = (v.neck_w + MAX_DELAY - n + (j * (n - 1)) / (m - 1)) % MAX_DELAY;
            *o = v.neck[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::measured_freq;

    fn run(bank: &mut Bank, p: &Params, n: usize) -> Vec<f32> {
        (0..n).map(|_| bank.tick(p)).collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    fn params() -> Params {
        Params {
            decay: 0.996,
            damping: 0.5,
            stiffness: 0.0,
            level: 1.0,
            couple: 0.0,
            bow_on: false,
            bow_speed: 0.0,
            bow_pressure: 0.0,
            bow_string: 0,
            drone: false,
        }
    }

    #[test]
    fn sympathetic_strings_ring_along() {
        // Octave pair: pluck the low string; the high one must start
        // moving through the shared bridge — and stay still without
        // coupling.
        let energy = |couple: f32| {
            let mut bank = Bank::new(&[110.0, 220.0]);
            let p = Params { couple, ..params() };
            bank.pluck(0, 0.2, &mut Rng(7));
            // Sympathy blooms slowly: the receiver integrates a trickle.
            let mut peak = 0.0f32;
            for _ in 0..8 {
                for _ in 0..11025 {
                    bank.tick(&p);
                }
                let mut shape = [0.0f32; 64];
                bank.shape(1, &mut shape);
                let rms = (shape.iter().map(|s| s * s).sum::<f32>() / 64.0).sqrt();
                peak = peak.max(rms);
            }
            peak
        };
        let coupled = energy(0.008);
        let isolated = energy(0.0);
        assert!(
            coupled > 0.005 && coupled > isolated * 10.0,
            "no sympathy: coupled {coupled} isolated {isolated}"
        );
    }

    #[test]
    fn sympathy_does_not_kill_sustain() {
        // The donated fraction comes out of the string's own reflection
        // every round trip, so an overscaled c makes every pluck
        // staccato. At the default scale a ring one second in must
        // still be most of what it would be uncoupled.
        let ring = |couple: f32| {
            let mut bank = Bank::new(&[220.0, 110.0]);
            let p = Params { couple, ..params() };
            bank.pluck(0, 0.2, &mut Rng(7));
            let out: Vec<f32> = (0..44100).map(|_| bank.tick(&p)).collect();
            rms(&out[22050..])
        };
        let with = ring(0.003);
        let without = ring(0.0);
        assert!(
            with > without * 0.4,
            "sympathy murders sustain: {with} vs {without}"
        );
    }

    #[test]
    fn coupling_stays_passive() {
        // Heavy coupling + a sustained bow: the redistributing bridge
        // must never let cross-string feedback run away.
        let mut bank = Bank::new(&[110.0, 220.0, 164.81]);
        let p = Params {
            couple: 0.5,
            bow_on: true,
            bow_speed: 0.6,
            bow_pressure: 0.5,
            ..params()
        };
        let out: Vec<f32> = (0..3 * 44100).map(|_| bank.tick(&p)).collect();
        assert!(out.iter().all(|s| s.is_finite()));
        let late = rms(&out[2 * 44100..]);
        assert!(late < 2.0, "coupled bank ran away: rms {late}");
    }

    #[test]
    fn pluck_rings_then_decays() {
        let mut bank = Bank::new(&[220.0]);
        bank.pluck(0, 0.2, &mut Rng(7));
        let out = run(&mut bank, &params(), 2 * 44100);
        assert!(out.iter().all(|s| s.is_finite()));
        let early = rms(&out[2205..11025]);
        let late = rms(&out[77175..]);
        assert!(early > 0.05, "pluck too quiet: {early}");
        assert!(
            late < early * 0.5,
            "pluck failed to decay: {early} -> {late}"
        );
        let f = measured_freq(&out[2205..44100], 220.0);
        let cents = 1200.0 * (f / 220.0).log2();
        assert!(
            cents.abs() < 10.0,
            "pluck pitch off: {f:.2} Hz ({cents:+.1}c)"
        );
    }

    #[test]
    fn bow_sustains_then_releases() {
        let mut bank = Bank::new(&[220.0]);
        let bowing = Params {
            bow_on: true,
            bow_speed: 0.6,
            bow_pressure: 0.7,
            ..params()
        };
        let out = run(&mut bank, &bowing, 2 * 44100);
        assert!(out.iter().all(|s| s.is_finite()));
        let sustained = rms(&out[66150..]);
        assert!(sustained > 0.05, "bow failed to sustain: rms {sustained}");
        // A steady bow reaches a steady state, not endless crescendo:
        // the last half-second must match the one before it.
        let earlier = rms(&out[44100..66150]);
        assert!(
            (sustained / earlier - 1.0).abs() < 0.15,
            "no steady state: {earlier} -> {sustained}"
        );
        // Lift the bow: the tone must die away.
        let tail = run(&mut bank, &params(), 2 * 44100);
        let released = rms(&tail[66150..]);
        assert!(
            released < sustained * 0.2,
            "string kept sounding after bow lift: {sustained} -> {released}"
        );
    }

    #[test]
    fn bow_locks_the_fundamental() {
        // Across the playable range, the string must speak at its own
        // pitch — not whistle a harmonic (the failure mode of both
        // single-loop bows and pinned bow points).
        for (speed, pressure) in [(0.3, 0.4), (0.5, 0.5), (0.6, 0.7), (0.9, 0.8)] {
            let mut bank = Bank::new(&[220.0]);
            let bowing = Params {
                bow_on: true,
                bow_speed: speed,
                bow_pressure: pressure,
                ..params()
            };
            let out = run(&mut bank, &bowing, 2 * 44100);
            // Global search: a harmonic lock shows up as a shorter
            // period. Octave guard: ac(2T) ≈ ac(T) for periodic signals,
            // so take the smallest near-maximal lag, not the argmax.
            let ac = |lag: usize| -> f32 {
                let n = 8192.min(out.len() - 44100 - lag);
                (0..n)
                    .map(|i| out[44100 + i] * out[44100 + i + lag])
                    .sum::<f32>()
                    / n as f32
            };
            let max = (40..=420).map(ac).fold(f32::MIN, f32::max);
            let best = (40..=420).find(|&l| ac(l) > 0.9 * max).unwrap();
            let coarse = SR / best as f32;
            assert!(
                (coarse / 220.0 - 1.0).abs() < 0.08,
                "speed {speed} pressure {pressure}: locked at {coarse:.0} Hz"
            );
            // With the fundamental confirmed dominant, measure it finely.
            let f = measured_freq(&out[44100..], 220.0);
            let cents = 1200.0 * (f / 220.0).log2();
            assert!(
                cents.abs() < 25.0,
                "speed {speed} pressure {pressure}: {f:.2} Hz ({cents:+.1}c)"
            );
        }
    }
}
