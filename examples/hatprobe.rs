//! Diagnostic: the hi-hat's sounds — open hit, closed hit, half-open
//! hit, foot chick, and what is left after the foot lifts — as rms in
//! windows after the event.
//!
//!     cargo run --release --example hatprobe
use std::sync::Arc;
use timber::analyze;
use timber::cymbal::Modes;
use timber::hihat::{HiHat, default_params};
use timber::util::{Rng, SR};
fn main() {
    let modes = Arc::new(Modes::compute());
    let run = |setup: &dyn Fn(&mut HiHat, &mut Rng)| -> Vec<f32> {
        let mut h = HiHat::new(modes.clone(), default_params());
        let mut rng = Rng(3);
        setup(&mut h, &mut rng);
        (0..(1.0 * SR) as usize).map(|_| h.tick(&mut rng)).collect()
    };
    let settle = |h: &mut HiHat, rng: &mut Rng, secs: f32| {
        for _ in 0..(secs * SR) as usize {
            h.tick(rng);
        }
    };
    let open = run(&|h, _| h.strike(1.0));
    let closed = run(&|h, rng| {
        h.pedal(true);
        settle(h, rng, 0.15);
        h.strike(1.0);
    });
    let half = run(&|h, rng| {
        let mut p = default_params();
        p.press = 0.9;
        h.set_params(p);
        h.pedal(true);
        settle(h, rng, 0.15);
        h.strike(1.0);
    });
    let chick = run(&|h, _| h.pedal(true));
    // Foot down, then up: what rings after the release (should be
    // next to nothing — a hat does not hum).
    let release = run(&|h, rng| {
        h.pedal(true);
        settle(h, rng, 0.2);
        h.pedal(false);
    });
    println!(
        "{:>8} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7}",
        "", "0-10ms", "10-50", "50-100", "100-300", "300-600", "peak"
    );
    for (name, b) in [
        ("open", &open),
        ("closed", &closed),
        ("half", &half),
        ("chick", &chick),
        ("release", &release),
    ] {
        let w = |a: f32, c: f32| analyze::rms(analyze::win(b, a, c));
        let peak = b.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        println!(
            "{name:>8} | {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} | {peak:>7.4}",
            w(0.0, 0.01),
            w(0.01, 0.05),
            w(0.05, 0.1),
            w(0.1, 0.3),
            w(0.3, 0.6)
        );
    }
}
