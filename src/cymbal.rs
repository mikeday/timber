//! Model 2e: the cymbal in modal coordinates — von Kármán done the way
//! the cymbal literature does it (Ducceschi & Touzé).
//!
//! The finite-difference plate (plate.rs) is the right object but the
//! wrong integrator for the nonlinearity: an explicit scheme with an
//! approximate stress solve leaks energy into single grid modes, and
//! the cure — an accurate stress solve every step — costs more than
//! realtime. So: take the plate's linear modes *once* from the grid,
//! and run the nonlinearity as what it is in that basis, a cubic
//! coupling between modes with fixed coefficients.
//!
//! Displacement w = Σ_k Φ_k q_k, Airy stress F = Σ_p Ψ_p η_p, with
//!   η_p = Σ_ij H^p_ij q_i q_j / ζ_p⁴,   H^p_ij = ∫ Ψ_p L(Φ_i, Φ_j),
//!   q̈_k = −ω_k² q_k − 2σ_k q̇_k − ν Σ_p η_p Σ_i H^p_ki q_i + stick.
//! That cubic force is the gradient of V = ¼ν Σ_p ζ_p⁴ η_p², so in the
//! continuum it conserves energy. The scheme integrates the linear
//! part exactly (each mode is an oscillator), each mode's *self*
//! coupling implicitly, and the cross coupling explicitly under a cap
//! on its matrix norm, with an energy budget as belt to those braces.
//! (A fully conservative implicit step — Bilbao's time-averaged form
//! solved by conjugate gradients — was tried and costs 3–4× realtime
//! per cymbal at this mode count; see step().)
//!
//! A structural gift: the Neumann Laplacian L and the bending
//! operator L² share eigenvectors, so the mode *shapes* don't depend
//! on stiffness or dome — only ω_k² = μ²K_k² + λ²K_k does. One
//! eigen-decomposition at startup; both knobs stay live.
//!
//! What it costs: the coupling tensor is sparse by symmetry (a disc's
//! angular selection rules), and the mode count sets the highest
//! frequency the cascade can reach — a plate has ~N modes below
//! N·(some Hz), so a realtime budget of ~100 modes tops out around
//! 2–3 kHz of wash on this grid. That is the honest price of realtime.

use std::sync::Arc;

use crate::grid::Disc;
use crate::plate::PlateParams;
use crate::stick::{self, Stick};
use crate::util::{Rng, SR};

pub type CymbalParams = PlateParams;

/// Disc radius in cells for the mode grid.
pub const R: usize = 10;
const W: usize = 2 * R + 5;
const CLAMP: f32 = 1.5;
/// Internal steps every DECIM samples: the modes kept reach ~3.3 kHz,
/// so 14.7 kHz internal loses nothing and the coupling costs a third
/// less.
const DECIM: usize = 3;
const SR_INT: f32 = SR / DECIM as f32;
/// Bending stiffness at `stiffness` = 1 (per-step² units). The exact
/// modal integrator has no CFL limit; the ceiling is that the
/// highest mode stays below Nyquist (K = 8 → ω = 2.8 rad/step).
const MU2_MAX: f32 = 0.12;
/// Modes kept, by frequency: the realtime budget. The k-th mode's
/// Laplacian eigenvalue is ≈ 8k/cells, so 120 of 340 reach K ≈ 2.8,
/// i.e. ~3.3 kHz at full stiffness — where the cascade tops out.
const N_MODES: usize = 120;
/// Stress modes kept: the coupling is weighted by 1/ζ_p⁴, so the
/// lowest few carry nearly all of it.
const N_STRESS: usize = 8;
/// Cross-coupling entries below this fraction of the largest cross
/// term are dropped: the tensor is sparse by the disc's angular
/// selection rules, and runtime cost is the entry count (~7k here,
/// ~0.3× realtime per ringing cymbal).
const H_KEEP: f32 = 0.05;
/// Cap on the coupling stiffness's matrix norm, Σ_p |ν η_p|·‖H^p‖ ≤
/// G_MAX (per-step² units against a linear ω² of up to ~8): the cross
/// terms are explicit, so this bounds how fast the cascade may run
/// and keeps hard hits saturating instead of exploding.
const G_MAX: f32 = 0.2;
/// Stick velocity per unit strength, per step.
const STICK_V: f32 = 0.012;
const ENERGY_SLACK: f32 = 1.05;

/// Everything about the disc that is fixed: mode shapes, the stress
/// modes, and the coupling between them. Expensive to build (two
/// eigen-decompositions of ~300×300 and the coupling sums); shared by
/// every cymbal.
pub struct Modes {
    disc: Disc,
    /// Mode shapes, N_MODES × cells, unit-norm.
    phi: Vec<Vec<f32>>,
    /// Laplacian eigenvalue of each mode (positive).
    k: Vec<f32>,
    /// Mean of each mode shape over the disc (what it radiates as a
    /// whole).
    phi_mean: Vec<f32>,
    /// Mode shape at the pickup.
    phi_pick: Vec<f32>,
    /// The von Kármán coupling, one sparse symmetric H^p per stress
    /// mode (raw entries; the 2 for off-diagonal pairs and 1/ζ_p⁴ are
    /// applied at runtime).
    coupling: Vec<Coupling>,
}

struct Coupling {
    /// 1/ζ_p⁴ for this stress mode.
    inv_zeta4: f32,
    /// Off-diagonal and diagonal entries of the symmetric H^p.
    entries: Vec<(u16, u16, f32)>,
    /// Frobenius norm of H^p — an upper bound on its spectral norm —
    /// for the drive cap: the explicit cross terms are stable only
    /// while the coupling stiffness's *matrix* norm stays small, and
    /// a cap on the largest entry did not bound that (it blew up).
    h_norm: f32,
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix (f64).
/// Returns eigenvalues ascending and eigenvectors as rows.
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..30 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        let norm: f64 = (0..n * n).map(|i| a[i] * a[i]).sum();
        if off < 1e-22 * norm.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));
    let vals = order.iter().map(|&i| a[i * n + i]).collect();
    let vecs = order
        .iter()
        .map(|&col| (0..n).map(|row| v[row * n + col]).collect())
        .collect();
    (vals, vecs)
}

impl Modes {
    pub fn compute() -> Self {
        Self::compute_with(N_MODES, N_STRESS, H_KEEP)
    }

    /// Mode budget, stress-mode budget, and the fraction of the
    /// largest *cross* coupling below which entries are dropped.
    pub fn compute_with(n_modes: usize, n_stress: usize, h_keep: f32) -> Self {
        let disc = Disc::new(R, 2, CLAMP);
        debug_assert_eq!(disc.w, W);
        let (mask, cells) = (&disc.mask, &disc.cells);
        let n = cells.len();
        let index: std::collections::HashMap<usize, usize> =
            cells.iter().enumerate().map(|(a, &b)| (b, a)).collect();
        let at = |idx: usize| index.get(&idx).copied();

        // Neumann Laplacian on the domain, Dirichlet at the clamp:
        // −L is positive; its eigenvectors are the plate's modes.
        let mut l = vec![0.0f64; n * n];
        for (a, &idx) in cells.iter().enumerate() {
            let mut cnt = 0.0;
            for nb in [idx - 1, idx + 1, idx - W, idx + W] {
                if mask[nb] {
                    cnt += 1.0;
                    if let Some(b) = at(nb) {
                        l[a * n + b] += 1.0;
                    }
                }
            }
            l[a * n + a] -= cnt;
        }
        let neg_l: Vec<f64> = l.iter().map(|x| -x).collect();
        let (kvals, kvecs) = jacobi(neg_l, n);

        // Clamped biharmonic (Φ = 0 outside the disc and at the clamp):
        // its eigenvectors are the stress modes.
        let mut b = vec![0.0f64; n * n];
        for (a, &idx) in cells.iter().enumerate() {
            b[a * n + a] = 20.0;
            for (nb, wgt) in [
                (idx - 1, -8.0),
                (idx + 1, -8.0),
                (idx - W, -8.0),
                (idx + W, -8.0),
                (idx - W - 1, 2.0),
                (idx - W + 1, 2.0),
                (idx + W - 1, 2.0),
                (idx + W + 1, 2.0),
                (idx - 2, 1.0),
                (idx + 2, 1.0),
                (idx - 2 * W, 1.0),
                (idx + 2 * W, 1.0),
            ] {
                if let Some(bb) = at(nb) {
                    b[a * n + bb] += wgt;
                }
            }
        }
        let (zvals, zvecs) = jacobi(b, n);

        // Keep the lowest N_MODES plate modes above the sub-audio
        // floor (K > tiny), as f32 fields on the grid.
        let keep: Vec<usize> = (0..n).filter(|&i| kvals[i] > 1e-4).take(n_modes).collect();
        let field = |vec: &Vec<f64>| {
            let mut f = vec![0.0f32; W * W];
            for (a, &idx) in cells.iter().enumerate() {
                f[idx] = vec[a] as f32;
            }
            f
        };
        let phi: Vec<Vec<f32>> = keep.iter().map(|&i| field(&kvecs[i])).collect();
        let k: Vec<f32> = keep.iter().map(|&i| kvals[i] as f32).collect();
        let phi_mean = phi
            .iter()
            .map(|f| cells.iter().map(|&c| f[c]).sum::<f32>() / n as f32)
            .collect();
        let pick = disc.at(0, (2 * R / 3) as isize);
        let phi_pick = phi.iter().map(|f| f[pick]).collect();

        // Coupling H^p_ij = Σ Ψ_p L(Φ_i, Φ_j), stress modes lowest first.
        let curv = |f: &[f32], idx: usize| -> (f32, f32, f32) {
            let g = |nb: usize| if mask[nb] { f[nb] } else { f[idx] };
            let fxx = g(idx - 1) + g(idx + 1) - 2.0 * f[idx];
            let fyy = g(idx - W) + g(idx + W) - 2.0 * f[idx];
            let d = |nb: usize| if mask[nb] { f[nb] } else { 0.0 };
            let fxy = 0.25 * (d(idx + W + 1) - d(idx + W - 1) - d(idx - W + 1) + d(idx - W - 1));
            (fxx, fyy, fxy)
        };
        let nm = phi.len();
        let curvs: Vec<Vec<(f32, f32, f32)>> = phi
            .iter()
            .map(|f| cells.iter().map(|&c| curv(f, c)).collect())
            .collect();
        let mut coupling = Vec::new();
        let ns = n_stress.min(n);
        for p in 0..ns {
            let psi: Vec<f32> = (0..n).map(|a| zvecs[p][a] as f32).collect();
            let mut h = vec![0.0f32; nm * nm];
            for i in 0..nm {
                for j in i..nm {
                    let mut s = 0.0f32;
                    for a in 0..n {
                        let (ixx, iyy, ixy) = curvs[i][a];
                        let (jxx, jyy, jxy) = curvs[j][a];
                        s += psi[a] * (ixx * jyy + iyy * jxx - 2.0 * ixy * jxy);
                    }
                    h[i * nm + j] = s;
                }
            }
            // Threshold against the largest cross term: the diagonal
            // (self-stiffening) entries are the biggest and always
            // kept, but it is the cross terms that move energy between
            // modes — the cascade — and a threshold set by the
            // diagonal quietly threw them away.
            let mut top = 0.0f32;
            for i in 0..nm {
                for j in i + 1..nm {
                    top = top.max(h[i * nm + j].abs());
                }
            }
            let mut entries = Vec::new();
            for i in 0..nm {
                for j in i..nm {
                    let v = h[i * nm + j];
                    if i == j || v.abs() > h_keep * top {
                        entries.push((i as u16, j as u16, v));
                    }
                }
            }
            let h_norm = entries
                .iter()
                .map(|&(i, j, v)| if i == j { v * v } else { 2.0 * v * v })
                .sum::<f32>()
                .sqrt();
            coupling.push(Coupling {
                inv_zeta4: 1.0 / zvals[p].max(1e-6) as f32,
                entries,
                h_norm,
            });
        }
        Modes {
            disc,
            phi,
            k,
            phi_mean,
            phi_pick,
            coupling,
        }
    }

    pub fn n_modes(&self) -> usize {
        self.phi.len()
    }

    pub fn n_cells(&self) -> usize {
        self.disc.cells.len()
    }

    pub fn coupling_entries(&self) -> usize {
        self.coupling.iter().map(|c| c.entries.len()).sum()
    }

    /// (cross entries, self entries).
    pub fn coupling_stats(&self) -> (usize, usize) {
        let mut cross = 0;
        let mut diag = 0;
        for c in &self.coupling {
            for &(i, j, _) in &c.entries {
                if i == j {
                    diag += 1;
                } else {
                    cross += 1;
                }
            }
        }
        (cross, diag)
    }
}

pub struct Cymbal {
    modes: Arc<Modes>,
    p: CymbalParams,
    q: Vec<f32>,
    q_prev: Vec<f32>,
    /// Per-mode integrator coefficients: 2cos(ωk)e^{−σ}, e^{−2σ},
    /// force gain 2(1−cos ωk)/ω², and ω².
    c1: Vec<f32>,
    c2: Vec<f32>,
    cf: Vec<f32>,
    w2: Vec<f32>,
    sigma: Vec<f32>,
    /// Mode shape summed over the contact patch.
    phi_patch: Vec<f32>,
    patch_mass: f32,
    stick: Stick,
    force: Vec<f32>,
    q_next: Vec<f32>,
    /// Each mode's own nonlinear stiffness this step, Σ_p g_p H^p_ii.
    dk: Vec<f32>,
    energy_max: f32,
    phase: usize,
    out_a: f32,
    out_b: f32,
    quiet: u32,
}

impl Cymbal {
    pub fn new(modes: Arc<Modes>, p: CymbalParams) -> Self {
        let n = modes.n_modes();
        let mut c = Cymbal {
            modes,
            p,
            q: vec![0.0; n],
            q_prev: vec![0.0; n],
            c1: vec![0.0; n],
            c2: vec![0.0; n],
            cf: vec![0.0; n],
            w2: vec![0.0; n],
            sigma: vec![0.0; n],
            phi_patch: vec![0.0; n],
            patch_mass: 1.0,
            stick: Stick::default(),
            force: vec![0.0; n],
            q_next: vec![0.0; n],
            dk: vec![0.0; n],
            energy_max: 0.0,
            phase: 0,
            out_a: 0.0,
            out_b: 0.0,
            quiet: u32::MAX / 2,
        };
        c.set_params(p);
        c
    }

    pub fn set_params(&mut self, p: CymbalParams) {
        self.p = p;
        let mu2 = p.stiffness.clamp(0.0, 1.0) * MU2_MAX;
        let lam = p.dome.max(0.0) * std::f32::consts::TAU * (R as f32 + 0.85) / (2.405 * SR_INT);
        let lam2 = lam * lam;
        let base = 1.0 / (p.decay.max(0.01) * SR_INT);
        for (i, &k) in self.modes.k.iter().enumerate() {
            let w2 = mu2 * k * k + lam2 * k;
            let w = w2.sqrt().min(0.95 * std::f32::consts::PI);
            let hz = w * SR_INT / std::f32::consts::TAU;
            // Loss rises with frequency squared: hf_damp 1 has a 1 kHz
            // partial dying twice as fast as the lowest, 5 kHz 26×.
            let sigma = base * (1.0 + p.hf_damp.clamp(0.0, 1.0) * (hz / 1000.0).powi(2));
            self.w2[i] = w * w;
            self.sigma[i] = sigma;
            self.c1[i] = 2.0 * w.cos() * (-sigma).exp();
            self.c2[i] = (-2.0 * sigma).exp();
            self.cf[i] = if w > 1e-3 {
                2.0 * (1.0 - w.cos()) / (w * w)
            } else {
                1.0
            };
        }
        let center = self.modes.disc.strike_center(p.strike_pos);
        let (patch, mass) =
            stick::patch(&self.modes.disc.mask, W, center, p.mallet.clamp(0.8, 5.0));
        self.patch_mass = mass;
        for (i, f) in self.modes.phi.iter().enumerate() {
            self.phi_patch[i] = patch.iter().map(|&(idx, w)| w * f[idx]).sum();
        }
    }

    fn under_stick(&self) -> f32 {
        self.phi_patch.iter().zip(&self.q).map(|(a, b)| a * b).sum()
    }

    pub fn strike(&mut self, strength: f32) {
        self.stick.throw(self.under_stick(), strength * STICK_V);
        self.quiet = 0;
    }

    fn step(&mut self) -> f32 {
        let n = self.q.len();
        let f = self
            .stick
            .step(self.under_stick(), self.p.hardness, self.patch_mass);
        for i in 0..n {
            self.force[i] = f * self.phi_patch[i];
        }
        // The cubic coupling. Its stiffness K(q_n) = ν Σ_p η_p H^p is
        // split: each mode's self term goes on the new state
        // (implicit — explicit, a hard hit on a ringing plate turned
        // momentary softening into exponential growth, which the
        // energy budget rendered as static), the cross terms — the
        // cascade — are explicit forces. Explicit cross terms are only
        // stable while ‖K‖ stays small against the step, so the drive
        // is capped by the matrix norm (see G_MAX). A conservative
        // implicit treatment (Bilbao's time-averaged form, solved by
        // conjugate gradients) was tried and costs 3–4× realtime per
        // cymbal on this budget; this is the affordable point.
        let nu = self.p.nonlin.max(0.0);
        let mut potential = 0.0f32;
        let n_stress = self.modes.coupling.len();
        for i in 0..n {
            self.dk[i] = 0.0;
        }
        if nu > 0.0 {
            let modes = &*self.modes;
            let q = &self.q;
            let force = &mut self.force;
            let dk = &mut self.dk;
            for cp in &modes.coupling {
                let mut eta = 0.0f32;
                for &(i, j, h) in &cp.entries {
                    // SAFETY: entries were built from indices < n.
                    let (qi, qj) =
                        unsafe { (*q.get_unchecked(i as usize), *q.get_unchecked(j as usize)) };
                    let t = h * qi * qj;
                    eta += if i == j { t } else { 2.0 * t };
                }
                let eta = eta * cp.inv_zeta4;
                let cap = G_MAX / (cp.h_norm * n_stress as f32);
                let g = (nu * eta).clamp(-cap, cap);
                // The potential the *applied* (capped) force derives
                // from: ¼ g η ζ⁴ is the quartic ¼νη²ζ⁴ when uncapped
                // and bounded when not. Booking the uncapped quartic
                // here let the budget carry fictitious energy the
                // kinetic side then grew into — a slow late swell.
                potential += 0.25 * g * eta / cp.inv_zeta4;
                for &(i, j, h) in &cp.entries {
                    let (i, j) = (i as usize, j as usize);
                    // SAFETY: as above.
                    unsafe {
                        if i == j {
                            *dk.get_unchecked_mut(i) += g * h;
                        } else {
                            *force.get_unchecked_mut(i) -= g * h * *q.get_unchecked(j);
                            *force.get_unchecked_mut(j) -= g * h * *q.get_unchecked(i);
                        }
                    }
                }
            }
        }
        for i in 0..n {
            let dk = self.dk[i].max(-0.5 * self.w2[i]);
            self.q_next[i] = (self.c1[i] * self.q[i] - self.c2[i] * self.q_prev[i]
                + self.cf[i] * self.force[i])
                / (1.0 + self.cf[i] * dk);
        }
        let mut energy = potential;
        let mut diss = 0.0f32;
        for i in 0..n {
            let qn = self.q_next[i];
            let v = 0.5 * (qn - self.q_prev[i]);
            energy += 0.5 * (v * v + self.w2[i] * self.q[i] * self.q[i]);
            diss += 2.0 * self.sigma[i] * v * v;
            self.q_prev[i] = self.q[i];
            self.q[i] = qn;
        }
        // Energy budget (see plate.rs): the coupling conserves energy
        // exactly in the continuum; the explicit step can drift.
        if !energy.is_finite() || energy > 1e12 {
            self.q.iter_mut().for_each(|x| *x = 0.0);
            self.q_prev.iter_mut().for_each(|x| *x = 0.0);
            self.energy_max = 0.0;
        } else if nu > 0.0 {
            if self.stick.on() {
                self.energy_max = self.energy_max.max(energy);
            } else {
                self.energy_max = (self.energy_max - diss).max(0.0);
                if energy > self.energy_max * ENERGY_SLACK && energy > 0.0 {
                    let k = (self.energy_max / energy).sqrt();
                    for i in 0..n {
                        self.q[i] *= k;
                        self.q_prev[i] *= k;
                    }
                }
            }
        }
        // Pickup: velocity at the point plus a little of the mean.
        let mut out = 0.0f32;
        for i in 0..n {
            let v = self.q[i] - self.q_prev[i];
            out += v * (self.modes.phi_pick[i] + 0.5 * self.modes.phi_mean[i]);
        }
        out * 100.0
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

/// The cymbals. Coupling is in modal units (see `nonlin`): hundreds
/// here is a real cymbal's nonlinearity at a normal hit.
pub fn default_kit() -> Vec<(&'static str, CymbalParams)> {
    vec![
        (
            "ride",
            CymbalParams {
                stiffness: 0.8,
                dome: 90.0,
                decay: 5.0,
                hf_damp: 0.3,
                strike_pos: 0.5,
                hardness: 0.95,
                mallet: 1.0,
                nonlin: 1000.0,
                level: 0.5,
            },
        ),
        (
            "crash",
            CymbalParams {
                stiffness: 0.5,
                dome: 60.0,
                decay: 3.0,
                hf_damp: 0.15,
                strike_pos: 0.85,
                hardness: 0.9,
                mallet: 1.2,
                nonlin: 4000.0,
                level: 0.5,
            },
        ),
    ]
}

pub struct Kit {
    pads: Vec<Cymbal>,
}

impl Kit {
    pub fn new(modes: Arc<Modes>, params: Vec<CymbalParams>) -> Self {
        Kit {
            pads: params
                .into_iter()
                .map(|p| Cymbal::new(modes.clone(), p))
                .collect(),
        }
    }
    pub fn set_params(&mut self, i: usize, p: CymbalParams) {
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

    fn modes() -> Arc<Modes> {
        Arc::new(Modes::compute())
    }

    fn render(modes: &Arc<Modes>, p: CymbalParams, strength: f32, secs: f32) -> Vec<f32> {
        let mut m = Cymbal::new(modes.clone(), p);
        let mut rng = Rng(3);
        m.strike(strength);
        (0..(secs * SR) as usize)
            .map(|_| m.tick(&mut rng))
            .collect()
    }

    #[test]
    fn jacobi_finds_eigenpairs() {
        // The eigen-solver underneath everything: on a small symmetric
        // matrix, A·v = λ·v for every pair, the vectors are
        // orthonormal, and the eigenvalues come out ascending.
        let n = 12;
        let mut rng = Rng(5);
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rng.next() as f64;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (vals, vecs) = jacobi(a.clone(), n);
        for k in 0..n {
            let av: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| a[i * n + j] * vecs[k][j]).sum())
                .collect();
            let err: f64 = (0..n)
                .map(|i| (av[i] - vals[k] * vecs[k][i]).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-9, "pair {k}: |Av − λv| = {err}");
            for l in 0..n {
                let dot: f64 = (0..n).map(|i| vecs[k][i] * vecs[l][i]).sum();
                let want = if k == l { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "vectors {k},{l} dot {dot}");
            }
            if k > 0 {
                assert!(vals[k] >= vals[k - 1], "not ascending at {k}");
            }
        }
        // And the disc's Laplacian modes are positive and ascending.
        let m = modes();
        assert!(m.k[0] > 0.0);
        assert!(m.k.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn modal_cymbal_washes_and_settles() {
        let modes = modes();
        assert!(modes.n_modes() >= 60, "{} modes", modes.n_modes());
        let crash = default_kit()
            .into_iter()
            .find(|(n, _)| *n == "crash")
            .unwrap()
            .1;
        let linear = render(
            &modes,
            CymbalParams {
                nonlin: 0.0,
                ..crash
            },
            1.6,
            2.0,
        );
        let wash = render(&modes, crash, 2.0, 2.0);
        assert!(wash.iter().all(|s| s.is_finite()));
        // Brightness must climb after the hit with the coupling on,
        // where the linear plate only fades.
        let c = |b: &[f32], t0: f32, t1: f32| analyze::centroid(analyze::win(b, t0, t1));
        let (l0, l1) = (c(&linear, 0.005, 0.02), c(&linear, 0.2, 0.4));
        let (w0, w1) = (c(&wash, 0.005, 0.02), c(&wash, 0.2, 0.4));
        assert!(
            w1 / w0 > 1.2 * (l1 / l0),
            "no wash: {w0:.0} -> {w1:.0} (linear {l0:.0} -> {l1:.0})"
        );
        // …and it dies away like a cymbal, not a hum: the last half
        // second is well below the first, and no single partial owns
        // the late ring.
        let early = analyze::rms(analyze::win(&wash, 0.0, 0.25));
        let late = analyze::rms(analyze::win(&wash, 1.5, 2.0));
        assert!(late < early * 0.5, "no decay: {early:.3} -> {late:.3}");
        let ring = analyze::win(&wash, 0.5, 1.0);
        let mut f = 100.0;
        let (mut total, mut top) = (0.0f32, 0.0f32);
        while f < 10000.0 {
            let m = analyze::mag(ring, f);
            total += m;
            top = top.max(m);
            f *= 1.02;
        }
        assert!(
            top < 0.25 * total,
            "one partial owns the ring: {:.0}%",
            100.0 * top / total
        );
    }
}
