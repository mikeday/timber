//! Shared plumbing: sample rate, noise, mixing.

pub const SAMPLE_RATE: u32 = 44_100;
pub const SR: f32 = SAMPLE_RATE as f32;

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
