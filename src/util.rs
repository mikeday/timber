//! Shared plumbing: sample rate, noise, mixing, thread-shared floats.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

pub const SAMPLE_RATE: u32 = 44_100;
pub const SR: f32 = SAMPLE_RATE as f32;

/// An f32 shared between the UI and audio threads without locking,
/// stored as bits in an AtomicU32.
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        AtomicF32(AtomicU32::new(v.to_bits()))
    }
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Relaxed))
    }
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Relaxed)
    }
}

/// Estimate the dominant frequency near `approx_freq` by autocorrelation
/// with parabolic sub-sample refinement. The tests' ear.
pub fn measured_freq(buf: &[f32], approx_freq: f32) -> f32 {
    let approx = SR / approx_freq;
    let ac = |lag: usize| -> f32 {
        let n = 8192.min(buf.len() - lag);
        (0..n).map(|i| buf[i] * buf[i + lag]).sum()
    };
    let (lo, hi) = ((approx * 0.94) as usize, (approx * 1.06) as usize + 1);
    let best = (lo..=hi).max_by(|&x, &y| ac(x).total_cmp(&ac(y))).unwrap();
    let (a, b, c) = (ac(best - 1), ac(best), ac(best + 1));
    SR / (best as f32 + 0.5 * (a - c) / (a - 2.0 * b + c))
}

/// Tiny deterministic xorshift so every render is reproducible.
/// Returns samples in -1.0..1.0.
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Mix a rendered note into the output buffer at a start time in seconds.
pub fn place(mix: &mut Vec<f32>, note: &[f32], at: f32) {
    let start = (at * SR) as usize;
    if mix.len() < start + note.len() {
        mix.resize(start + note.len(), 0.0);
    }
    for (i, s) in note.iter().enumerate() {
        mix[start + i] += s;
    }
}

/// Fade the last 10ms of a buffer to zero. A render whose duration ends
/// while the model is still vibrating would otherwise step straight to
/// silence — an audible click, worst at low frequencies.
pub fn fade_out(buf: &mut [f32]) {
    let n = ((0.01 * SR) as usize).min(buf.len());
    let len = buf.len();
    for i in 0..n {
        buf[len - n + i] *= 1.0 - (i + 1) as f32 / n as f32;
    }
}

/// Scale a buffer so its peak is `level`. Lets each model's `level` field
/// mean the same thing regardless of how hot the raw synthesis runs.
pub fn normalize(buf: &mut [f32], level: f32) {
    let peak = buf.iter().fold(0.0f32, |m, s| m.max(s.abs())).max(1e-9);
    for s in buf.iter_mut() {
        *s *= level / peak;
    }
}
