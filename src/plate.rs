//! Model 2d: the cymbal as a plate — the membrane grid with bending.
//!
//! (The desk's cymbals run the modal formulation in cymbal.rs, which
//! takes its mode shapes from this same disc; this finite-difference
//! plate stays as the reference model and for `analyze plate:…`.)
//!
//! A thin plate resists curvature, not just stretch:
//!
//!   u_tt = −κ²∇⁴u + c²∇²u − 2σ₀u_t + 2σ₁∇²u_t
//!
//! The biharmonic term makes a mode's frequency go as its wavenumber
//! *squared*, so the partials spread from a few tens of Hz to the top
//! of the band, hundreds of them, nothing harmonic about their ratios
//! — the sound of struck metal. The edge is free (a cymbal's is) and
//! the center is held, as on a stand.
//!
//! The c² term is not in a textbook plate; it stands in for
//! curvature. A flat bronze disc this size would have its lowest mode
//! around 20 Hz; a cymbal's bow and bell stiffen the low modes without
//! touching the high ones, and a membrane-like tension does exactly
//! that. So: `stiffness` sets the bending (the high end, the metal),
//! `dome` sets the low end.
//!
//! And the wash. A plate bent out of its plane must stretch in it,
//! and the in-plane stress feeds back on the bending — von Kármán's
//! coupling. Per step: the Airy stress function Φ from ∇⁴Φ = −L(u,u),
//! then a force L(Φ,u) on the plate, where L is the bracket
//! f_xx·g_yy + f_yy·g_xx − 2·f_xy·g_xy. That cubic term is what pours
//! the energy a stick left in a few low modes into thousands of high
//! ones over the first ~100 ms: the crash's shimmer arrives *after*
//! the hit, from the plate itself. Φ is solved by Gauss–Seidel sweeps
//! on the 13-point biharmonic, warm-started from the last step (the
//! stress changes slowly against the step) with Φ = 0 outside the
//! disc, which is the stress-free edge.

use crate::grid::Disc;
use crate::stick::{self, Stick};
use crate::util::{Rng, SR};

/// Internal steps happen every DECIM samples (interpolated between):
/// the plate's band reaches ~9 kHz at its stiffest, and the stress
/// solve is the most expensive thing in the desk.
const DECIM: usize = 2;
const SR_INT: f32 = SR / DECIM as f32;

/// Disc radius in cells.
pub const R: usize = 10;
/// Grid width: two cells of border for the 13-point biharmonic.
const W: usize = 2 * R + 5;
/// Center clamp radius in cells: the stand. Also pins the tilt
/// (a free plate's zero-frequency rigid modes).
const CLAMP: f32 = 1.5;
/// Stability of the combined scheme at the grid's corner wavenumber:
/// λ² + 8μ² ≤ 1/2 (Laplacian eigenvalue −8, biharmonic 64).
const MU2_MAX: f32 = 0.06;
/// Stick velocity per unit strength, per step.
const STICK_V: f32 = 0.012;
/// Frequency-dependent loss ceiling (see mesh.rs S1_MAX).
const S1_MAX: f32 = 0.03;
/// Successive over-relaxation sweeps per step for the Airy stress
/// function. Accuracy here is not optional: with a rough Φ the
/// explicit scheme leaks energy into one grid mode and the crash
/// decays into a high hum; 40 plain Gauss–Seidel sweeps cured it,
/// and over-relaxation reaches that in these few.
const SWEEPS: usize = 5;
const SOR: f32 = 1.7;
/// Cap on the in-plane stress (each component of nonlin·Φ's second
/// differences). The stress acts on the plate exactly like the
/// membrane's tension, so it eats the same CFL budget: λ² + 8μ² +
/// 8·stress ≤ 1/2. Capping the *force* did not help — a force cap
/// leaves the effective stiffness unbounded at the grid's corner
/// wavenumber, and hard hits still ran away.
const STRESS_SHARE: f32 = 0.05;
/// Displacement past which the scheme has come apart anyway: reset
/// rather than shriek.
const BLOWUP: f32 = 50.0;
/// Slack on the energy budget before the state is scaled back.
const ENERGY_SLACK: f32 = 1.02;

#[derive(Clone, Copy)]
pub struct PlateParams {
    /// Bending stiffness, 0..1 of the grid's stable maximum. Higher =
    /// thicker/stiffer metal: the partials spread wider and the top
    /// end rises.
    pub stiffness: f32,
    /// Curvature stand-in, as the fundamental (Hz) the disc would have
    /// from that tension alone. Raises the low modes.
    pub dome: f32,
    /// Overall decay time constant, seconds.
    pub decay: f32,
    /// Frequency-dependent loss, 0..1: how much faster the high
    /// partials die (a cymbal's "wash" fading to its "bell").
    pub hf_damp: f32,
    /// Where the stick lands, 0 (near the bell) to 1 (the edge).
    pub strike_pos: f32,
    /// Stick hardness and footprint, as for the membrane.
    pub hardness: f32,
    pub mallet: f32,
    /// Von Kármán coupling strength, 0 = linear plate. Physically the
    /// ratio of strike displacement to plate thickness: thin cymbals
    /// hit hard are very nonlinear. The scale is the model's: single
    /// digits for this finite-difference plate, thousands for the
    /// modal cymbal (cymbal.rs), whose default kit sets it.
    pub nonlin: f32,
    pub level: f32,
}

pub struct Plate {
    p: PlateParams,
    u: Vec<f32>,
    u_prev: Vec<f32>,
    u_next: Vec<f32>,
    lap: Vec<f32>,
    lapp: Vec<f32>,
    /// Airy stress function and the bracket −L(u,u) driving it, and
    /// the plate's second differences (u_xx, u_yy, u_xy) this step.
    phi: Vec<f32>,
    rhs: Vec<f32>,
    curv: Vec<(f32, f32, f32)>,
    disc: Disc,
    /// Neighbor count in the domain, for the free-edge Laplacian.
    nbrs: Vec<u8>,
    lam2: f32,
    mu2: f32,
    s0: f32,
    s1: f32,
    /// What the in-plane stress may take of the CFL budget.
    stress_cap: f32,
    /// Energy budget: the true von Kármán plate conserves energy, so
    /// with the stick gone the plate's energy may only fall. The
    /// explicit scheme leaks a slow growth at strong coupling; any
    /// energy above the budget is scaled off, which leaves the
    /// cascade — the *distribution* across modes — untouched.
    energy_max: f32,
    patch: Vec<(usize, f32)>,
    patch_mass: f32,
    stick: Stick,
    pick: usize,
    phase: usize,
    phase_stress: bool,
    out_a: f32,
    out_b: f32,
    quiet: u32,
}

impl Plate {
    pub fn new(p: PlateParams) -> Self {
        let disc = Disc::new(R, 2, CLAMP);
        debug_assert_eq!(disc.w, W);
        let mask = &disc.mask;
        let nbrs = (0..W * W)
            .map(|idx| {
                if !mask[idx] {
                    return 0;
                }
                [idx - 1, idx + 1, idx - W, idx + W]
                    .iter()
                    .filter(|&&n| mask[n])
                    .count() as u8
            })
            .collect();
        // Pickup: off the strike radius (which runs along +x), two
        // thirds out — where a mic or an ear sits.
        let pick = disc.at(0, (2 * R / 3) as isize);
        let mut m = Plate {
            p,
            u: vec![0.0; W * W],
            u_prev: vec![0.0; W * W],
            u_next: vec![0.0; W * W],
            lap: vec![0.0; W * W],
            lapp: vec![0.0; W * W],
            phi: vec![0.0; W * W],
            rhs: vec![0.0; W * W],
            curv: vec![(0.0, 0.0, 0.0); W * W],
            disc,
            nbrs,
            lam2: 0.0,
            mu2: 0.0,
            s0: 0.0,
            s1: 0.0,
            stress_cap: 0.0,
            energy_max: 0.0,
            patch: Vec::new(),
            patch_mass: 1.0,
            stick: Stick::default(),
            pick,
            phase: 0,
            phase_stress: true,
            out_a: 0.0,
            out_b: 0.0,
            quiet: u32::MAX / 2,
        };
        m.set_params(p);
        m
    }

    pub fn set_params(&mut self, p: PlateParams) {
        self.p = p;
        self.mu2 = p.stiffness.clamp(0.0, 1.0) * MU2_MAX;
        // Tension as for the membrane: λ from the fundamental a disc
        // of this radius would have.
        let lam = p.dome.max(0.0) * std::f32::consts::TAU * (R as f32 + 0.85) / (2.405 * SR_INT);
        self.lam2 = (lam * lam).min(0.5 - 8.0 * self.mu2).max(0.0);
        self.stress_cap = STRESS_SHARE * (0.5 - self.lam2 - 8.0 * self.mu2).max(0.0) / 8.0;
        let rate = 1.0 / (p.decay.max(0.01) * SR_INT);
        self.s0 = rate;
        // The σ₁ term acts on the Laplacian, whose eigenvalue reaches 8
        // at the grid corner: a partial at the top of the band loses
        // 8·σ₁ per step. hf_damp 1 puts its decay near 60 ms.
        self.s1 = (p.hf_damp.clamp(0.0, 1.0) * 2.5e-4).min(S1_MAX);
        let center = self.disc.strike_center(p.strike_pos);
        let (patch, mass) = stick::patch(&self.disc.mask, W, center, p.mallet.clamp(0.8, 5.0));
        self.patch = patch;
        self.patch_mass = mass;
    }

    fn under_stick(&self) -> f32 {
        self.patch.iter().map(|&(i, w)| w * self.u[i]).sum()
    }

    pub fn contact_steps(&self) -> u32 {
        self.stick.contact_steps()
    }

    pub fn strike(&mut self, strength: f32) {
        self.stick.throw(self.under_stick(), strength * STICK_V);
        self.quiet = 0;
    }

    /// Free-edge Laplacian: missing neighbors contribute nothing (zero
    /// normal slope), so the edge can move.
    fn laplacian(&self, src: &[f32], dst: &mut [f32]) {
        for j in 2..W - 2 {
            for i in 2..W - 2 {
                let idx = j * W + i;
                if !self.disc.mask[idx] {
                    continue;
                }
                let mut s = 0.0;
                for n in [idx - 1, idx + 1, idx - W, idx + W] {
                    if self.disc.mask[n] {
                        s += src[n];
                    }
                }
                dst[idx] = s - self.nbrs[idx] as f32 * src[idx];
            }
        }
    }

    /// Second differences at a cell, Neumann-filled at the edge.
    #[inline]
    fn curvatures(&self, f: &[f32], idx: usize) -> (f32, f32, f32) {
        let m = &self.disc.mask;
        let g = |n: usize| if m[n] { f[n] } else { f[idx] };
        let fxx = g(idx - 1) + g(idx + 1) - 2.0 * f[idx];
        let fyy = g(idx - W) + g(idx + W) - 2.0 * f[idx];
        let d = |n: usize| if m[n] { f[n] } else { 0.0 };
        let fxy = 0.25 * (d(idx + W + 1) - d(idx + W - 1) - d(idx - W + 1) + d(idx - W - 1));
        (fxx, fyy, fxy)
    }

    /// The Airy stress function for the current bending: solve
    /// ∇⁴Φ = −L(u,u) on the disc with Φ = 0 outside (stress-free
    /// edge — so Φ's differences need no edge rule), a few over-
    /// relaxed Gauss–Seidel sweeps from last step's Φ.
    fn stress(&mut self) {
        let mut rhs = std::mem::take(&mut self.rhs);
        let mut curv = std::mem::take(&mut self.curv);
        for &idx in &self.disc.cells {
            let (uxx, uyy, uxy) = self.curvatures(&self.u, idx);
            curv[idx] = (uxx, uyy, uxy);
            rhs[idx] = -2.0 * (uxx * uyy - uxy * uxy);
        }
        let mut phi = std::mem::take(&mut self.phi);
        for _ in 0..SWEEPS {
            for &idx in &self.disc.cells {
                let p = &phi;
                let near = p[idx - 1] + p[idx + 1] + p[idx - W] + p[idx + W];
                let diag = p[idx - W - 1] + p[idx - W + 1] + p[idx + W - 1] + p[idx + W + 1];
                let far = p[idx - 2] + p[idx + 2] + p[idx - 2 * W] + p[idx + 2 * W];
                let gs = (rhs[idx] + 8.0 * near - 2.0 * diag - far) / 20.0;
                phi[idx] += SOR * (gs - phi[idx]);
            }
        }
        self.phi = phi;
        self.rhs = rhs;
        self.curv = curv;
    }

    fn step(&mut self) -> f32 {
        let force = self
            .stick
            .step(self.under_stick(), self.p.hardness, self.patch_mass);
        // Laplacian of u and of u_prev (for the σ₁ term), then the
        // biharmonic as the Laplacian of the Laplacian, same edge rule.
        let mut lap = std::mem::take(&mut self.lap);
        let mut lapp = std::mem::take(&mut self.lapp);
        self.laplacian(&self.u, &mut lap);
        self.laplacian(&self.u_prev, &mut lapp);
        // The stress changes at the pace of u², slower than u: solving
        // it every other step halves the cost.
        let nonlin = self.p.nonlin.max(0.0);
        if nonlin > 0.0 && self.phase_stress {
            self.stress();
        }
        self.phase_stress = !self.phase_stress;
        let (lam2, mu2, s0, s1) = (self.lam2, self.mu2, self.s0, self.s1);
        let mut blown = false;
        for &idx in &self.disc.cells {
            let mut b = 0.0;
            for n in [idx - 1, idx + 1, idx - W, idx + W] {
                if self.disc.mask[n] {
                    b += lap[n];
                }
            }
            b -= self.nbrs[idx] as f32 * lap[idx];
            // The membrane-like force from in-plane stress: L(Φ,u),
            // with each stress component capped (see STRESS_SHARE).
            let nl = if nonlin > 0.0 {
                let (uxx, uyy, uxy) = self.curv[idx];
                let p = &self.phi;
                let cap = self.stress_cap;
                let sxx = (nonlin * (p[idx - 1] + p[idx + 1] - 2.0 * p[idx])).clamp(-cap, cap);
                let syy = (nonlin * (p[idx - W] + p[idx + W] - 2.0 * p[idx])).clamp(-cap, cap);
                let sxy = (nonlin
                    * 0.25
                    * (p[idx + W + 1] - p[idx + W - 1] - p[idx - W + 1] + p[idx - W - 1]))
                    .clamp(-cap, cap);
                sxx * uyy + syy * uxx - 2.0 * sxy * uxy
            } else {
                0.0
            };
            let un = (2.0 * self.u[idx] - (1.0 - s0) * self.u_prev[idx] + lam2 * lap[idx]
                - mu2 * b
                + s1 * (lap[idx] - lapp[idx])
                + nl)
                / (1.0 + s0);
            // Written to catch NaN as well as a runaway.
            blown |= !un.is_finite() || un.abs() >= BLOWUP;
            self.u_next[idx] = un;
        }
        // Energy after the update (kinetic + bending + tension, in
        // consistent per-step units), against the budget.
        if nonlin > 0.0 {
            let (mut e, mut diss) = (0.0f32, 0.0f32);
            for &idx in &self.disc.cells {
                let v = self.u_next[idx] - self.u[idx];
                e += v * v + mu2 * lap[idx] * lap[idx] - lam2 * self.u[idx] * lap[idx];
                // What the damping terms took this step: σ₀ on the
                // velocity, σ₁ on the velocity's Laplacian.
                let v2 = self.u_next[idx] - self.u_prev[idx];
                diss += 0.5 * (s0 * v2 * v2 - s1 * (lap[idx] - lapp[idx]) * v2);
            }
            if self.stick.on() {
                self.energy_max = self.energy_max.max(e);
            } else {
                // The budget falls by the measured dissipation and at
                // least at the nominal σ₀ rate: energy the scheme
                // conjures cannot sit at the budget and hum.
                self.energy_max =
                    (self.energy_max - diss.max(0.0)).min(self.energy_max * (1.0 - 2.0 * s0));
                if e > self.energy_max * ENERGY_SLACK && e > 0.0 {
                    let k = (self.energy_max / e).sqrt();
                    for &idx in &self.disc.cells {
                        self.u_next[idx] *= k;
                        self.u[idx] *= k;
                    }
                }
            }
        }
        self.lap = lap;
        self.lapp = lapp;
        if blown {
            for v in [
                &mut self.u,
                &mut self.u_prev,
                &mut self.u_next,
                &mut self.phi,
            ] {
                v.iter_mut().for_each(|x| *x = 0.0);
            }
            self.energy_max = 0.0;
        }
        if force > 0.0 {
            for &(idx, w) in &self.patch {
                self.u_next[idx] += force * w / (1.0 + s0);
            }
        }
        std::mem::swap(&mut self.u_prev, &mut self.u);
        std::mem::swap(&mut self.u, &mut self.u_next);
        // Pickup: local velocity at a point plus a little of the mean
        // (the low modes radiate as a whole).
        let mean: f32 = self
            .disc
            .cells
            .iter()
            .map(|&i| self.u[i] - self.u_prev[i])
            .sum::<f32>()
            / self.disc.cells.len() as f32;
        (self.u[self.pick] - self.u_prev[self.pick] + 0.5 * mean) * 100.0
    }

    pub fn tick(&mut self, _rng: &mut Rng) -> f32 {
        if self.quiet > (3.0 * SR) as u32 {
            return 0.0;
        }
        if self.phase == 0 {
            self.out_a = self.out_b;
            self.out_b = self.step();
            if self.out_b.abs() < 1e-5 {
                self.quiet += DECIM as u32;
            } else {
                self.quiet = 0;
            }
        }
        let t = self.phase as f32 / DECIM as f32;
        self.phase = (self.phase + 1) % DECIM;
        (self.out_a + (self.out_b - self.out_a) * t) * self.p.level
    }
}

/// The cymbals, calibrated by ear so far.
pub fn default_kit() -> Vec<(&'static str, PlateParams)> {
    vec![
        (
            "ride",
            PlateParams {
                stiffness: 0.7,
                dome: 90.0,
                decay: 4.0,
                hf_damp: 0.3,
                strike_pos: 0.5,
                hardness: 0.95,
                mallet: 1.0,
                nonlin: 2.0,
                level: 0.5,
            },
        ),
        (
            "crash",
            PlateParams {
                stiffness: 0.35,
                dome: 60.0,
                decay: 2.5,
                hf_damp: 0.15,
                strike_pos: 0.85,
                hardness: 0.9,
                mallet: 1.2,
                nonlin: 5.0,
                level: 0.5,
            },
        ),
    ]
}

pub struct Kit {
    pads: Vec<Plate>,
}

impl Kit {
    pub fn new(params: Vec<PlateParams>) -> Self {
        Kit {
            pads: params.into_iter().map(Plate::new).collect(),
        }
    }
    pub fn set_params(&mut self, i: usize, p: PlateParams) {
        if let Some(m) = self.pads.get_mut(i) {
            m.set_params(p);
        }
    }
    pub fn strike(&mut self, i: usize, strength: f32) {
        if let Some(m) = self.pads.get_mut(i) {
            m.strike(strength);
        }
    }
    pub fn tick(&mut self, rng: &mut Rng) -> f32 {
        self.pads.iter_mut().map(|m| m.tick(rng)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze;

    /// The ride, linear: these tests are about the plate's modes.
    fn ride() -> PlateParams {
        PlateParams {
            nonlin: 0.0,
            ..default_kit()[0].1
        }
    }

    fn render(p: PlateParams, strength: f32, secs: f32) -> Vec<f32> {
        let mut m = Plate::new(p);
        let mut rng = Rng(3);
        m.strike(strength);
        (0..(secs * SR) as usize)
            .map(|_| m.tick(&mut rng))
            .collect()
    }

    /// Count spectral peaks above a floor between lo and hi Hz.
    fn partials(buf: &[f32], lo: f32, hi: f32) -> usize {
        let step = 5.0;
        let bands: Vec<f32> = (0..)
            .map(|k| lo + k as f32 * step)
            .take_while(|&f| f <= hi)
            .map(|f| analyze::mag(buf, f))
            .collect();
        let top = bands.iter().cloned().fold(0.0, f32::max);
        (1..bands.len() - 1)
            .filter(|&i| {
                bands[i] > bands[i - 1] && bands[i] >= bands[i + 1] && bands[i] > top * 0.003
            })
            .count()
    }

    #[test]
    fn plate_is_stable_dense_and_decays() {
        let out = render(ride(), 1.0, 3.0);
        assert!(out.iter().all(|s| s.is_finite()));
        let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!((0.05..3.0).contains(&peak), "peak {peak}");
        // Metal: many partials, spread over the band.
        let n = partials(&out[2205..46305], 50.0, 8000.0);
        assert!(n > 20, "only {n} partials — that is not a plate");
        let early = analyze::rms(&out[..4410]);
        let late = analyze::rms(&out[110250..114660]);
        assert!(late < early * 0.5, "no decay: {early} -> {late}");
    }

    #[test]
    fn stiffness_and_dome_push_the_spectrum() {
        // Bending stiffness spreads the partials upward: the highest
        // partial the plate still carries (−40 dB of its strongest,
        // over the ring) sits higher on a stiffer plate. Dome raises
        // the lowest mode. (The centroid is the wrong probe: a stiffer
        // plate pushes partials past what the stick can excite, and
        // the centroid *falls* — stage 2's cascade is what fills that
        // in on a real cymbal.)
        let edge = |p: PlateParams| {
            let out = render(p, 1.0, 1.2);
            let ring = &out[2205..46305];
            let mut f = 100.0;
            let mut top = 0.0f32;
            let mut edge = 0.0;
            let mut mags = Vec::new();
            while f < 10000.0 {
                let m = analyze::mag(ring, f);
                top = top.max(m);
                mags.push((f, m));
                f *= 1.03;
            }
            for (f, m) in mags {
                if m > top * 0.01 {
                    edge = f;
                }
            }
            edge
        };
        let thin = edge(PlateParams {
            stiffness: 0.2,
            ..ride()
        });
        let thick = edge(PlateParams {
            stiffness: 0.9,
            ..ride()
        });
        assert!(
            thick > thin * 1.15,
            "stiffness: thin {thin:.0} thick {thick:.0}"
        );
        // The lowest partial the plate carries (peaks above −30 dB of
        // the strongest in 30..600 Hz), not the strongest: the dome
        // moves the bottom of the spectrum, not its center of weight.
        let lo = |p: PlateParams| {
            let ring = &render(p, 1.0, 1.2)[2205..46305];
            let bands: Vec<(f32, f32)> = (0..)
                .map(|k| 30.0 + k as f32 * 2.0)
                .take_while(|&f| f <= 600.0)
                .map(|f| (f, analyze::mag(ring, f)))
                .collect();
            let top = bands.iter().map(|b| b.1).fold(0.0, f32::max);
            (1..bands.len() - 1)
                .find(|&i| {
                    bands[i].1 > bands[i - 1].1
                        && bands[i].1 >= bands[i + 1].1
                        && bands[i].1 > top * 0.03
                })
                .map(|i| bands[i].0)
                .unwrap_or(0.0)
        };
        let flat = lo(PlateParams {
            dome: 30.0,
            ..ride()
        });
        let domed = lo(PlateParams {
            dome: 200.0,
            ..ride()
        });
        assert!(domed > flat * 1.3, "dome: flat {flat:.0} domed {domed:.0}");
    }
}
