//! Diagnostic: the steel pan pad's note beside a recording — the
//! analyzer's report for the pad at a given pitch.
//!
//!     cargo run --release --example panprobe -- 233
use timber::analyze;
use timber::drums::{Kit, PadParams};
use timber::modal::PAN;
use timber::util::{Rng, SR};
fn main() {
    let f: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(261.63);
    // Mirrors the desk's "steel pan" pad.
    let p = PadParams {
        modes: PAN,
        freq: 261.63,
        glide: 0.02,
        glide_time: 0.06,
        damp: 1.0,
        noise: 1.0,
        noise_decay: 0.004,
        noise_tone: 0.0,
        bloom: 0.5,
        bloom_delay: 0.012,
        bloom_spread: 0.03,
        drive: 0.0,
        shimmer: 0.25,
        attack: 0.0,
        claps: 1,
        tremolo: 0.0,
        bloom_lo: 1.0,
        bleed: 0.08,
        roll_rate: 14.0,
        roll_strength: 0.75,
        damper: false,
        level: 1.0,
        choke: None,
    };
    let mut kit = Kit::new(vec![p]);
    let mut rng = Rng(3);
    kit.strike_note(0, f);
    let out: Vec<f32> = (0..(2.5 * SR) as usize)
        .map(|_| kit.tick(&mut rng))
        .collect();
    let r = analyze::report("pad", &analyze::normalize(out), Some(f));
    analyze::print_side_by_side(&[r]);
}
