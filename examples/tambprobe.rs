//! Diagnostic: the tambourine pad as the analyzer sees it, at the
//! desk's settings (mirrored here), beside samples/tamb_*.wav.
//!
//!     cargo run --release --example tambprobe
use timber::analyze;
use timber::drums::{Kit, PadParams};
use timber::modal::TAMBOURINE;
use timber::util::{Rng, SR};
fn main() {
    let p = PadParams {
        modes: TAMBOURINE,
        freq: 3800.0,
        glide: 0.0,
        glide_time: 0.1,
        damp: 1.0,
        noise: 1.0,
        noise_decay: 0.035,
        noise_tone: 0.1,
        bloom: 0.0,
        bloom_delay: 0.025,
        bloom_spread: 0.0,
        drive: 0.0,
        shimmer: 0.5,
        attack: 0.0,
        claps: 2,
        tremolo: 0.0,
        bloom_lo: 0.0,
        bleed: 0.0,
        roll_rate: 14.0,
        roll_strength: 0.75,
        damper: false,
        level: 1.0,
        choke: None,
    };
    let mut kit = Kit::new(vec![p]);
    let mut rng = Rng(3);
    kit.strike(0);
    let out: Vec<f32> = (0..(1.5 * SR) as usize)
        .map(|_| kit.tick(&mut rng))
        .collect();
    let r = analyze::report("pad", &analyze::normalize(out), Some(2400.0));
    analyze::print_side_by_side(&[r]);
}
