//! Measuring a drum hit — ours or a recording of a real one — in the
//! terms the mesh is built from: how the level falls, how bright the
//! attack is and how that brightness fades, how the pitch bends in,
//! and which modes ring for how long. The same report for both is
//! what makes a recording a calibration target rather than a vibe.
//!
//! Everything is band-scanned DFT magnitudes (no FFT dependency; a
//! drum hit is short and the resolution we want is a few Hz).

use std::f32::consts::TAU;

use crate::util::SR;

/// Hann-windowed magnitude at one frequency over a slice.
pub fn mag(buf: &[f32], f: f32) -> f32 {
    let w = TAU * f / SR;
    let n = buf.len() as f32;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (i, s) in buf.iter().enumerate() {
        let win = 0.5 - 0.5 * (TAU * i as f32 / n).cos();
        re += s * win * (w * i as f32).cos();
        im += s * win * (w * i as f32).sin();
    }
    (re * re + im * im).sqrt() / n
}

pub fn rms(b: &[f32]) -> f32 {
    (b.iter().map(|s| s * s).sum::<f32>() / b.len().max(1) as f32).sqrt()
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// Spectral centroid, 40 Hz..8 kHz on a 4% grid.
pub fn centroid(buf: &[f32]) -> f32 {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    let mut f = 40.0;
    while f < 8000.0 {
        let m = mag(buf, f);
        num += f * m;
        den += m;
        f *= 1.04;
    }
    num / den.max(1e-9)
}

/// Frequency of the strongest component in a band.
pub fn peak_in(buf: &[f32], lo: f32, hi: f32, step: f32) -> f32 {
    let mut best = (lo, 0.0f32);
    let mut f = lo;
    while f <= hi {
        let m = mag(buf, f);
        if m > best.1 {
            best = (f, m);
        }
        f += step;
    }
    best.0
}

/// Slice by time, clamped to the buffer.
pub fn win(buf: &[f32], t0: f32, t1: f32) -> &[f32] {
    let a = ((t0 * SR) as usize).min(buf.len());
    let b = ((t1 * SR) as usize).min(buf.len());
    &buf[a..b]
}

/// One ringing component: frequency, magnitude, decay time constant.
#[derive(Clone, Copy, Debug)]
pub struct ModePeak {
    pub freq: f32,
    pub mag: f32,
    pub tau: f32,
}

/// Spectral peaks of the ring (50..850 ms) between 0.5·f0 and 12·f0,
/// strongest eight, with a decay time constant fitted over four
/// 200 ms windows from 250 ms on — after any pitch glide has landed,
/// or a fixed-frequency probe sees a mode *arrive* and calls it
/// growth.
pub fn modes(buf: &[f32], f0: f32) -> Vec<ModePeak> {
    let ring = win(buf, 0.05, 0.85);
    let step = 2.0;
    let bands: Vec<(f32, f32)> = (0..)
        .map(|k| 0.5 * f0 + k as f32 * step)
        .take_while(|&f| f <= 12.0 * f0)
        .map(|f| (f, mag(ring, f)))
        .collect();
    if bands.len() < 5 {
        return Vec::new();
    }
    let mut pk: Vec<(f32, f32)> = (2..bands.len() - 2)
        .filter(|&i| {
            bands[i].1 > bands[i - 1].1
                && bands[i].1 >= bands[i + 1].1
                && bands[i].1 > bands[i - 2].1
                && bands[i].1 > bands[i + 2].1
        })
        .map(|i| bands[i])
        .collect();
    pk.sort_by(|a, b| b.1.total_cmp(&a.1));
    pk.truncate(8);
    // Drop the noise floor: a recording's silence has "peaks" too.
    let floor = pk.first().map_or(0.0, |p| p.1 * 0.001);
    pk.retain(|p| p.1 > floor);
    pk.sort_by(|a, b| a.0.total_cmp(&b.0));
    pk.iter()
        .map(|&(f, m)| {
            let ts = [0.25, 0.45, 0.65, 0.85];
            let ys: Vec<f32> = ts
                .iter()
                .map(|&t| mag(win(buf, t, t + 0.2), f).max(1e-9).ln())
                .collect();
            let tm = 0.55 + 0.1;
            let ym = ys.iter().sum::<f32>() / 4.0;
            let num: f32 = ts
                .iter()
                .zip(&ys)
                .map(|(t, y)| (t + 0.1 - tm) * (y - ym))
                .sum();
            let den: f32 = ts.iter().map(|t| (t + 0.1 - tm).powi(2)).sum();
            let slope = num / den;
            ModePeak {
                freq: f,
                mag: m,
                tau: if slope < 0.0 {
                    -1.0 / slope
                } else {
                    f32::INFINITY
                },
            }
        })
        .collect()
}

/// The envelope windows every report uses, seconds.
pub const WINDOWS: [(f32, f32); 8] = [
    (0.0, 0.005),
    (0.005, 0.02),
    (0.02, 0.05),
    (0.05, 0.1),
    (0.1, 0.2),
    (0.2, 0.4),
    (0.4, 0.8),
    (0.8, 1.4),
];

/// Everything measured about one hit.
pub struct Report {
    pub name: String,
    pub f0: f32,
    /// dB rel. the loudest window, per WINDOWS.
    pub envelope: Vec<f32>,
    /// rms per 1 ms for the first 12 ms, dB rel. the loudest.
    pub attack: Vec<f32>,
    /// Spectral centroid per window (first six).
    pub brightness: Vec<f32>,
    /// (time, Hz) of the fundamental.
    pub pitch: Vec<(f32, f32)>,
    pub modes: Vec<ModePeak>,
}

/// Scale to peak 1 so a recording at any level compares.
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let peak = v.iter().fold(0.0f32, |a, s| a.max(s.abs())).max(1e-9);
    v.iter_mut().for_each(|s| *s /= peak);
    v
}

/// Trim leading silence: the hit starts where the signal first
/// exceeds −30 dB of its peak, less a millisecond of run-up.
pub fn trim_onset(v: &[f32]) -> &[f32] {
    let peak = v.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    let thresh = peak * 0.0316;
    let at = v.iter().position(|s| s.abs() > thresh).unwrap_or(0);
    &v[at.saturating_sub(44)..]
}

/// The fundamental: the strongest component below 400 Hz in the
/// settled ring (50..450 ms), refined to 0.5 Hz.
pub fn find_f0(buf: &[f32]) -> f32 {
    let ring = win(buf, 0.05, 0.45);
    let coarse = peak_in(ring, 30.0, 400.0, 2.0);
    peak_in(ring, coarse - 3.0, coarse + 3.0, 0.5)
}

pub fn report(name: &str, buf: &[f32], f0: Option<f32>) -> Report {
    let f0 = f0.unwrap_or_else(|| find_f0(buf));
    let e: Vec<f32> = WINDOWS
        .iter()
        .map(|&(t0, t1)| rms(win(buf, t0, t1)))
        .collect();
    let emax = e.iter().cloned().fold(0.0, f32::max);
    let envelope = e.iter().map(|&x| db(x / emax)).collect();
    let a: Vec<f32> = (0..12)
        .map(|k| rms(win(buf, k as f32 * 0.001, (k + 1) as f32 * 0.001)))
        .collect();
    let amax = a.iter().cloned().fold(0.0, f32::max);
    let attack = a.iter().map(|&x| db(x / amax)).collect();
    let brightness = WINDOWS[..6]
        .iter()
        .map(|&(t0, t1)| centroid(win(buf, t0, t1)))
        .collect();
    // Pitch windows long enough to resolve a low fundamental.
    let pw = (4.0 / f0).max(0.03);
    let pitch = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
        .iter()
        .map(|&t| (t, peak_in(win(buf, t, t + pw), 0.7 * f0, 1.45 * f0, 0.5)))
        .collect();
    Report {
        name: name.into(),
        f0,
        envelope,
        attack,
        brightness,
        pitch,
        modes: modes(buf, f0),
    }
}

/// Print reports side by side.
pub fn print_side_by_side(reports: &[Report]) {
    let head = |label: &str| {
        print!("  {label:>10}");
        for r in reports {
            print!(" {:>9}", r.name);
        }
        println!();
    };
    println!("attack — rms per 1 ms (dB rel. loudest ms):");
    head("ms");
    for k in 0..12 {
        print!("  {k:>10}");
        for r in reports {
            print!(" {:>9.1}", r.attack[k]);
        }
        println!();
    }
    println!("\nenvelope (dB rel. loudest window):");
    head("window");
    for (i, &(t0, t1)) in WINDOWS.iter().enumerate() {
        print!("  {:>4.0}-{:<5.0}", t0 * 1000.0, t1 * 1000.0);
        for r in reports {
            print!(" {:>9.1}", r.envelope[i]);
        }
        println!();
    }
    println!("\nbrightness — spectral centroid (Hz) per window:");
    head("window");
    for (i, &(t0, t1)) in WINDOWS[..6].iter().enumerate() {
        print!("  {:>4.0}-{:<5.0}", t0 * 1000.0, t1 * 1000.0);
        for r in reports {
            print!(" {:>9.0}", r.brightness[i]);
        }
        println!();
    }
    println!("\nfundamental trajectory (Hz):");
    head("at");
    for i in 0..6 {
        print!("  {:>7.0}ms ", reports[0].pitch[i].0 * 1000.0);
        for r in reports {
            print!(" {:>9.1}", r.pitch[i].1);
        }
        println!();
    }
    for r in reports {
        println!(
            "\n{} modes (f0 {:.1} Hz; ring from 50 ms; level rel. strongest; decay τ):",
            r.name, r.f0
        );
        let top = r.modes.iter().map(|m| m.mag).fold(0.0, f32::max);
        for m in &r.modes {
            println!(
                "  {:>7.1} Hz  x{:<5.2} {:>6.1} dB  τ {:>5.2} s",
                m.freq,
                m.freq / r.f0,
                db(m.mag / top),
                m.tau
            );
        }
    }
}

/// A first reading of mesh parameters off a report: the ones that map
/// directly. The rest (tension, hardness, air) are matched by ear
/// against the numbers this same report prints for the mesh.
pub fn suggest(r: &Report) {
    println!("\nmesh reading:");
    println!("  freq        {:.1}", r.f0);
    let f1 = r
        .modes
        .iter()
        .min_by(|a, b| (a.freq - r.f0).abs().total_cmp(&(b.freq - r.f0).abs()));
    if let Some(f1) = f1 {
        if f1.tau.is_finite() {
            println!("  decay       {:.2}   (fundamental τ)", f1.tau);
        }
        // Overtone damping from the strongest overtone's decay:
        // rate ratio = 1 + h·((f/f0)² − 1).
        let over = r
            .modes
            .iter()
            .filter(|m| m.freq > r.f0 * 1.3 && m.tau.is_finite() && m.tau > 0.0)
            .max_by(|a, b| a.mag.total_cmp(&b.mag));
        if let (Some(o), true) = (over, f1.tau.is_finite()) {
            let ratio = f1.tau / o.tau;
            let h = (ratio - 1.0) / ((o.freq / r.f0).powi(2) - 1.0);
            println!(
                "  overtone damp {:.2}   (x{:.2} mode rings {:.2} s vs {:.2} s)",
                h.clamp(0.0, 0.95),
                o.freq / r.f0,
                o.tau,
                f1.tau
            );
        }
        if let Some(m11) = r
            .modes
            .iter()
            .find(|m| ((m.freq / r.f0) / 1.59 - 1.0).abs() < 0.06)
        {
            println!(
                "  (1,1) mode  {:.1} dB   (strike pos: center ≈ −25, 0.3 ≈ −15, 0.5 ≈ −8)",
                db(m11.mag / f1.mag)
            );
        }
    }
    let bend = r.pitch[0].1 / r.pitch[5].1 - 1.0;
    println!(
        "  bend        {:+.1}%  onset vs settled  (tension: raise until the mesh bends this much)",
        bend * 100.0
    );
    println!(
        "  attack      {:.0} Hz centroid in the first 5 ms  (hardness/click: match by ear)",
        r.brightness[0]
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Mesh, MeshParams};
    use crate::util::Rng;

    #[test]
    fn report_reads_our_own_drum_back() {
        // Render a mesh tom, hand the audio to the analyzer with no
        // hints: it must find the fundamental, the (1,1) mode, and a
        // decay near the one the drum was given.
        let p = MeshParams {
            freq: 140.0,
            decay: 0.5,
            hf_damp: 0.3,
            tension: 0.0,
            strike_pos: 0.4,
            rattle: 0.0,
            click: 0.0,
            drive: 0.0,
            air: 0.0,
            reso_freq: 50.0,
            reso_decay: 0.5,
            hardness: 0.7,
            mallet: 1.5,
            level: 1.0,
        };
        let mut m = Mesh::new(p);
        let mut rng = Rng(3);
        m.strike(1.0);
        let out: Vec<f32> = (0..(1.5 * SR) as usize).map(|_| m.tick(&mut rng)).collect();
        let hit = normalize(trim_onset(&out).to_vec());
        let r = report("tom", &hit, None);
        assert!((r.f0 / 140.0 - 1.0).abs() < 0.03, "f0 {}", r.f0);
        let fund = r
            .modes
            .iter()
            .min_by(|a, b| (a.freq - r.f0).abs().total_cmp(&(b.freq - r.f0).abs()))
            .unwrap();
        assert!(
            (0.35..0.7).contains(&fund.tau),
            "fundamental τ {} (drum given 0.5)",
            fund.tau
        );
        assert!(
            r.modes
                .iter()
                .any(|m| ((m.freq / r.f0) / 1.59 - 1.0).abs() < 0.06),
            "no (1,1): {:?}",
            r.modes
        );
    }
}
