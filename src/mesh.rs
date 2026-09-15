//! Model 2c: the membrane as a membrane — a 2D finite-difference mesh.
//!
//! modal.rs *types in* a drum's Bessel-function mode ratios. Here they
//! fall out: a disc of cells obeying the wave equation, struck at a
//! point, rings at whatever modes that disc has. Everything the modal
//! kit had to be told — the inharmonic ratios, which modes a center
//! hit excites, the tension pitch-glide — is a consequence here:
//!
//! - u_tt = c²∇²u − 2σ₀ u_t + 2σ₁ ∇²u_t, on a circular mask with a
//!   fixed rim. σ₀ is the head's overall decay, σ₁ makes high modes die
//!   first (as on a real head).
//! - Tension modulation: the wave speed rises with the membrane's
//!   strain energy, so a hard hit starts sharp and droops as it
//!   settles — the tom glide, emergent, strike-dependent.
//! - The stick is a stick: a mass thrown at the head through a Hertz
//!   contact spring. Contact duration emerges — and shrinks as the hit
//!   gets harder, so hard hits crack and soft hits thud, the way a real
//!   drum's brightness follows the player's arm.
//! - The snare rattle is noise gated by the head's actual velocity:
//!   the wires buzz only while the membrane moves.
//! - A kick is two heads and a shell of air. The air is a spring on
//!   the batter head's mean displacement, coupling it to a resonant
//!   head (one lumped mode); the batter dumps its energy into them
//!   fast — a quick pitch settle — and the resonant head rings out the
//!   deep tail. That fast-bend-slow-ring is the kick's "oomph", and a
//!   lone head cannot make it: its pitch and its ring are one energy.
//!
//! This grid is also the foundation the nonlinear cymbal plate will
//! stand on: swap the spatial operator for a biharmonic (stiffness) and
//! add the in-plane coupling, and the same disc becomes thin metal.
//!
//! The mesh steps at half the sample rate (interpolated between steps):
//! the bandwidth a drum head needs for a quarter of the cost.

use crate::grid::Disc;
use crate::stick::{self, Stick};
use crate::util::{Rng, SR};

/// Disc radius in cells.
pub const R: usize = 12;
/// Grid width including a one-cell fixed border.
const W: usize = 2 * R + 3;
/// Internal steps happen every DECIM samples.
const DECIM: usize = 2;
const SR_INT: f32 = SR / DECIM as f32;
/// Stability: λ² ≤ 1/2 for the 2D scheme, and the frequency-dependent
/// damping eats into it (λ² + 4·s1 ≤ 1/2). Cap both with margin —
/// tension modulation pushes λ² toward this ceiling on hard hits.
const LAM2_MAX: f32 = 0.34;
/// The staircase rim sits ~half a cell outside R, so the disc rings a
/// little flat of the formula; this effective radius is measured.
const R_EFF: f32 = R as f32 + 0.85;
/// Laplacian eigenvalue of the fundamental on this grid, (2.405/R)².
const K1: f32 = (2.405 / R_EFF) * (2.405 / R_EFF);
const S1_MAX: f32 = 0.03;
/// Energy → tension scale, chosen so `tension` ≈ 8 gives a normal
/// (strength 0.8) stick hit ~2 semitones of droop and a hard one
/// (1.6) ~5 — the range recorded toms show.
const ENERGY_GAIN: f32 = 0.6;
/// Most the tension can raise c² (+1.0 → +41% pitch, ~6 semitones).
/// Recorded toms bend 25–40% onset-to-settled (samples/), so the
/// earlier +26% cap was clipping real behavior; safe now that the
/// energy proxy is ripple-free (see ENERGY_SMOOTH).
const TENSION_MAX: f32 = 1.0;
/// Energy-proxy smoothing per step (two poles, ~35 Hz). The proxy is
/// the head's total energy — strain plus kinetic — which for every
/// mode is constant over a cycle. A proxy that ripples at 2f (mean u²
/// did) drives the wave speed at 2f, which is exactly the parametric-
/// resonance condition (Mathieu): a hard tom hit *grew* into a
/// permanent hum at the tension cap. The smoothing mops up what
/// dispersion and damping leave of the ripple.
const ENERGY_SMOOTH: f32 = 0.01;
/// Stick velocity per unit strength, in fundamental radians per step
/// (the same normalization as the pickup, so levels sit with the old
/// velocity-bump strike).
const STICK_V: f32 = 1.0;
/// Head-velocity envelope below which the snare wires stop rattling
/// (in output units: ~−15 dB below a strength-1 hit's swing).
const RATTLE_KNEE: f32 = 0.04;
/// Snare-wire resonance: a broad peak around 2.5 kHz (a recorded dry
/// snare's first 50 ms center at 1.1–1.6 kHz with the head still in
/// the mix, samples/). White noise straight through, first-differenced, was a
/// +6 dB/octave tilt to the top of the band — harsh, not wiry.
const RATTLE_HZ: f32 = 2500.0;
const RATTLE_R: f32 = 0.85;
/// The batter's (0,1) mode against a uniform pressure: with the mode
/// shape J₀ normalized to peak 1, its mean over the disc is 0.432 and
/// its mean square 0.27, so a uniform force f per cell drives the
/// modal coordinate by f·0.432/0.27 = 1.6·f, and the mode's own mean
/// displacement is 0.432·q. The two-head eigenproblem below is built
/// from these.
const J0_MEAN: f32 = 0.432;
const J0_MEAN_SQ: f32 = 0.27;

/// The pitch the two coupled heads actually play. Batter (0,1) mode q
/// with bare ω_b², resonant head x with spring kr, air spring ka on
/// (0.432·q + x):
///   q̈ = −(ω_b² + 1.6·0.432·ka) q − 1.6·ka·x
///   ẍ = −(kr + ka) x − 0.432·ka·q
/// Returns the eigenvalues (ω²) of the batter-dominant and the
/// resonant-head-dominant modes.
fn coupled_omega2(wb2: f32, ka: f32, kr: f32) -> (f32, f32) {
    let g = J0_MEAN / J0_MEAN_SQ;
    let (m00, m01, m10, m11) = (wb2 + g * J0_MEAN * ka, g * ka, J0_MEAN * ka, kr + ka);
    let tr = m00 + m11;
    let det = m00 * m11 - m01 * m10;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let (lo, hi) = (0.5 * (tr - disc), 0.5 * (tr + disc));
    // Batter participation: |x/q| for each eigenvector, compared in
    // mean-displacement terms; the mode where the batter moves more
    // is the pitch.
    let ratio = |l: f32| ((l - m00) / m01.max(1e-12)).abs() / J0_MEAN;
    if ratio(lo) <= ratio(hi) {
        (lo, hi)
    } else {
        (hi, lo)
    }
}

#[derive(Clone, Copy)]
pub struct MeshParams {
    /// Fundamental in Hz. Sets the wave speed (Courant number).
    pub freq: f32,
    /// Fundamental's decay time constant, seconds.
    pub decay: f32,
    /// How much faster overtones die than the fundamental, 0..0.95:
    /// a mode at f loses at the fundamental's rate × (1 + hf_damp·
    /// ((f/f₀)² − 1)). 0 = every mode rings as long as the fundamental;
    /// ~0.3 is the law the hand-tuned modal kit follows.
    pub hf_damp: f32,
    /// Tension modulation strength: pitch rise per unit strain energy.
    pub tension: f32,
    /// Where the stick lands, 0 (center: only the round modes) to 1
    /// (near the rim: everything).
    pub strike_pos: f32,
    /// Snare-wire rattle amount, gated by head velocity.
    pub rattle: f32,
    /// Beater click: the landing's own noise, a burst at the moment of
    /// contact — a tick from a wood tip, the front edge of a thud from
    /// felt (the burst lasts longer the softer the beater).
    pub click: f32,
    /// Soft saturation on the head's output, 0 = clean. Stands in for
    /// what the shell and the air inside do to a hard hit: harmonics
    /// that read as weight rather than volume. Same curve as the modal
    /// kit's drive.
    pub drive: f32,
    /// Air-cavity coupling to the resonant head, 0 = a single open
    /// head. Stiffness of the enclosed air relative to the head's own.
    /// The played pitch is held by a two-head eigen-solution that is
    /// accurate up to ~0.3; beyond that the coupled pair drifts sharp.
    pub air: f32,
    /// Resonant head fundamental, Hz, and its decay, seconds.
    pub reso_freq: f32,
    pub reso_decay: f32,
    /// Stick hardness, 0 (felt beater) .. 1 (wood tip): contact
    /// stiffness, hence contact time, hence brightness.
    pub hardness: f32,
    /// Contact footprint radius in cells: a stick tip ~1, a felt
    /// beater ~3.
    pub mallet: f32,
    pub level: f32,
}

pub struct Mesh {
    p: MeshParams,
    u: Vec<f32>,
    u_prev: Vec<f32>,
    u_next: Vec<f32>,
    mask: Vec<bool>,
    /// Active cell indices (inside the disc), for a tight update loop.
    cells: Vec<usize>,
    lam2: f32,
    s0: f32,
    s1: f32,
    phase: usize,
    out_a: f32,
    out_b: f32,
    /// Smoothed head energy (strain + kinetic, in fundamental-
    /// amplitude² units) — the proxy driving tension. Two-stage
    /// smoothing, tracking instantly while the stick is on (see step()).
    energy_raw: f32,
    energy: f32,
    rattle_lp: f32,
    /// The wires' own resonance, a two-pole bandpass on the noise.
    rattle_y1: f32,
    rattle_y2: f32,
    /// Resonant head: one lumped mode, displacement now and before,
    /// and its spring (solved with the air, see set_params).
    reso: f32,
    reso_prev: f32,
    reso_k: f32,
    /// Click burst envelope, set by the strike, decaying per step.
    click_env: f32,
    /// One-pole lowpass state for the click: felt is dull, wood bright.
    click_lp: f32,
    /// Contact patch: (cell, weight), weights summing to 1.
    patch: Vec<(usize, f32)>,
    /// Sum of the raw bump — the patch's mass in cell masses.
    patch_mass: f32,
    stick: Stick,
    /// Samples of near-silence so far; a silent head stops stepping.
    quiet: u32,
}

impl Mesh {
    pub fn new(p: MeshParams) -> Self {
        let Disc { w, mask, cells, .. } = Disc::new(R, 1, 0.0);
        debug_assert_eq!(w, W);
        let mut m = Mesh {
            p,
            u: vec![0.0; W * W],
            u_prev: vec![0.0; W * W],
            u_next: vec![0.0; W * W],
            mask,
            cells,
            lam2: 0.0,
            s0: 0.0,
            s1: 0.0,
            phase: 0,
            out_a: 0.0,
            out_b: 0.0,
            energy_raw: 0.0,
            energy: 0.0,
            rattle_lp: 0.0,
            rattle_y1: 0.0,
            rattle_y2: 0.0,
            reso: 0.0,
            reso_prev: 0.0,
            reso_k: 0.0,
            click_env: 0.0,
            click_lp: 0.0,
            patch: Vec::new(),
            patch_mass: 1.0,
            stick: Stick::default(),
            quiet: u32::MAX / 2,
        };
        m.set_params(p);
        m
    }

    pub fn set_params(&mut self, p: MeshParams) {
        self.p = p;
        // Circular membrane: f₁ = 2.405·c/(2πR). With λ = c·k/h and
        // R = R_cells·h, λ = f₁·2π·R_cells/(2.405·SR_int).
        let lam = p.freq * std::f32::consts::TAU * R_EFF / (2.405 * SR_INT);
        // Tuned *with* the air in the shell: `freq` is what the drum
        // plays at, so the bare head sits lower by the air's share —
        // found by bisection on the two-head eigenproblem. (Only the
        // mean-bearing (0,n) modes feel the air; the rest drop with
        // the bare head, as on a real drum.)
        // Both springs are solved together: the air pushes the
        // resonant head's mode around just as much.
        let target = lam * lam;
        self.lam2 = target.min(LAM2_MAX);
        let wr2 = (std::f32::consts::TAU * p.reso_freq.max(10.0) / SR_INT).powi(2);
        self.reso_k = wr2;
        if p.air > 0.0 {
            for _ in 0..4 {
                let kr = self.reso_k;
                let (mut lo, mut hi) = (0.05 * target, target);
                for _ in 0..24 {
                    let mid = 0.5 * (lo + hi);
                    let ka = p.air * mid * K1;
                    if coupled_omega2(mid * K1, ka, kr).0 < target * K1 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                self.lam2 = (0.5 * (lo + hi)).min(LAM2_MAX);
                let ka = p.air * self.lam2 * K1;
                let (mut lo, mut hi) = (0.05 * wr2, wr2);
                for _ in 0..24 {
                    let mid = 0.5 * (lo + hi);
                    if coupled_omega2(self.lam2 * K1, ka, mid).1 < wr2 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                self.reso_k = 0.5 * (lo + hi);
            }
        }
        // Split the fundamental's loss rate between the flat term σ₀
        // and the ∇²-weighted term σ₁ (rate σ₁·K per step for a mode
        // with Laplacian eigenvalue K): the fundamental always decays
        // in `decay`, overtones faster by the f² law in proportion to
        // hf_damp. Expressed relative to the fundamental so the knob
        // means the same thing at every pitch and every decay.
        let h = p.hf_damp.clamp(0.0, 0.95);
        let rate = 1.0 / (p.decay.max(0.01) * SR_INT);
        self.s0 = rate * (1.0 - h);
        // ×2: σ₀ acts on the two-step difference u_next − u_prev, σ₁
        // on the one-step lap − lap_prev, so per unit of u_t it is
        // worth half as much.
        self.s1 = (2.0 * rate * h / K1).min(S1_MAX);
        self.build_patch();
    }

    /// The contact footprint: a raised-cosine patch at the strike
    /// position (along one radius from the center).
    fn build_patch(&mut self) {
        let c = (R + 1) as f32;
        let cx = c + self.p.strike_pos.clamp(0.0, 1.0) * (R as f32 - 2.5);
        let (patch, mass) = stick::patch(&self.mask, W, (cx, c), self.p.mallet.clamp(0.8, 5.0));
        self.patch = patch;
        self.patch_mass = mass;
    }

    /// How long the last stick stayed on the head, in internal steps.
    pub fn contact_steps(&self) -> u32 {
        self.stick.contact_steps()
    }

    /// Head displacement under the stick.
    fn under_stick(&self) -> f32 {
        self.patch.iter().map(|&(i, w)| w * self.u[i]).sum()
    }

    /// Radians per internal step at the fundamental — the scale that
    /// relates a velocity kick to the displacement it produces.
    fn omega_k(&self) -> f32 {
        (std::f32::consts::TAU * self.p.freq / SR_INT).max(1e-4)
    }

    /// Strike: throw the stick at the head. Velocity is scaled by ω·k
    /// so the swing it produces is unit-order at any pitch — a fixed
    /// kick swings a low head enormously (displacement ~ v/ω), which
    /// floods the tension feedback.
    /// The head right now, as a W×W field (border included).
    pub fn surface(&self) -> &[f32] {
        &self.u
    }

    pub fn width() -> usize {
        W
    }

    pub fn strike(&mut self, strength: f32) {
        self.stick
            .throw(self.under_stick(), strength * self.omega_k() * STICK_V);
        self.click_env = self.p.click * strength;
        self.quiet = 0;
    }

    fn step(&mut self, rng: &mut Rng) -> f32 {
        // Tension modulation: the effective wave speed rises with the
        // head's strain energy — a hard hit is sharp, then droops.
        // Capped: an unbounded time-varying wave speed pumps the grid
        // modes (parametric gain beats the damping) — a hard hit would
        // blow up into a self-sustaining 400 Hz shriek.
        let stretch = (self.p.tension * ENERGY_GAIN * self.energy).min(TENSION_MAX);
        let lam2 = (self.lam2 * (1.0 + stretch)).min(LAM2_MAX);
        let (s0, s1) = (self.s0, self.s1);
        let force = self
            .stick
            .step(self.under_stick(), self.p.hardness, self.patch_mass);
        // Air spring: cavity pressure from the batter's mean inward
        // displacement plus the resonant head's, pushing both back.
        let air = if self.p.air > 0.0 {
            let mean_u =
                self.cells.iter().map(|&i| self.u[i]).sum::<f32>() / self.cells.len() as f32;
            let ka = self.p.air * self.lam2 * K1;
            ka * (mean_u + self.reso)
        } else {
            0.0
        };
        let (mut strain, mut kinetic, mut mono) = (0.0f32, 0.0f32, 0.0f32);
        for &idx in &self.cells {
            let u = &self.u;
            let up = &self.u_prev;
            let lap = u[idx - 1] + u[idx + 1] + u[idx - W] + u[idx + W] - 4.0 * u[idx];
            let lapp = up[idx - 1] + up[idx + 1] + up[idx - W] + up[idx + W] - 4.0 * up[idx];
            let un = (2.0 * u[idx] - (1.0 - s0) * up[idx] + lam2 * lap + s1 * (lap - lapp) - air)
                / (1.0 + s0);
            self.u_next[idx] = un;
            // Σ|∇u|² = −Σ u·∇²u on a fixed rim; velocity by the
            // centered difference.
            strain -= u[idx] * lap;
            kinetic += (un - up[idx]) * (un - up[idx]);
            mono += un - u[idx];
        }
        let mono = mono / self.cells.len() as f32;
        // The resonant head: a lumped mode driven by the same pressure
        // (its whole area, its whole mass — per unit mass the same
        // push as one batter cell).
        let mut reso_v = 0.0;
        if self.p.air > 0.0 {
            let kr = self.reso_k;
            let sr = 1.0 / (self.p.reso_decay.max(0.01) * SR_INT);
            let xn =
                (2.0 * self.reso - (1.0 - sr) * self.reso_prev - kr * self.reso - air) / (1.0 + sr);
            reso_v = xn - self.reso;
            self.reso_prev = self.reso;
            self.reso = xn;
        }
        if force > 0.0 {
            for &(idx, w) in &self.patch {
                self.u_next[idx] += force * w / (1.0 + s0);
            }
        }
        // Kinetic energy in the same units as strain: ω² = c²·K for a
        // grid mode, so u_t²/c² pairs with K·u². Normalized by the
        // fundamental's K so a pure fundamental of amplitude a reads
        // ≈ a²·(mean mode shape²).
        // Divided by the *current* (stretched) c²: normalizing by the
        // rest value leaves a ripple proportional to the stretch — and
        // the pump came back on hard hits.
        let e = (strain + 0.25 * kinetic / lam2) / (K1 * self.cells.len() as f32);
        if self.stick.on() {
            // The stick is the onset: no lag while it is pushing, so the
            // pitch peaks with the hit and droops from there.
            self.energy_raw = e;
            self.energy = e;
        } else {
            self.energy_raw += ENERGY_SMOOTH * (e - self.energy_raw);
            self.energy += ENERGY_SMOOTH * (self.energy_raw - self.energy);
        }
        // Rotate buffers.
        std::mem::swap(&mut self.u_prev, &mut self.u);
        std::mem::swap(&mut self.u, &mut self.u_next);

        // Pickup: what a listener hears is what the head radiates. At
        // wavelengths longer than the head that is its mean velocity
        // (a monopole), which the round modes dominate; the (1,1) mode
        // is a dipole and radiates weakly — a point pickup near the
        // strike heard it at nearly the fundamental's level. But a
        // close mic hears the near field, and recorded toms (samples/)
        // carry their overtones at −10..−16 dB with the (1,1) at −15:
        // the local term is what supplies that richness.
        // Normalized by ω·k: displacement-scale output, so a kick and
        // a tom struck alike are alike in level.
        let pick = (R + 1) * W + (R + 1) + R / 3;
        let local = self.u[pick] - self.u_prev[pick];
        // The resonant head radiates too; on a recorded tom its mode
        // sits −11..−16 dB under the batter's, on a kick it is most
        // of the tail.
        let v = (2.0 * mono + 0.6 * local + 0.6 * reso_v) / self.omega_k() * 0.8;

        // Snare wires: noise gated by how hard the head is moving,
        // with a knee — below a certain swing the wires stay pressed
        // to the head and stop rattling, so the buzz dies before the
        // ring does (a snare's buzz is a front-of-note thing; gated
        // linearly it followed the ring all the way down).
        let mut out = v;
        if self.p.rattle > 0.0 {
            self.rattle_lp += 0.02 * (v.abs() - self.rattle_lp);
            let drive = (self.rattle_lp - RATTLE_KNEE).max(0.0);
            let th = std::f32::consts::TAU * RATTLE_HZ / SR_INT;
            let y = rng.next() + 2.0 * RATTLE_R * th.cos() * self.rattle_y1
                - RATTLE_R * RATTLE_R * self.rattle_y2;
            self.rattle_y2 = self.rattle_y1;
            self.rattle_y1 = y;
            out += y * self.p.rattle * drive * 3.0;
        }
        // Beater click: 0.5 ms of noise from a wood tip, ~3 ms from
        // felt. (Scaling noise by the contact force's rise was the
        // first idea — physically neat, inaudible: a felt beater's
        // force rises over 8 ms, so per step it rises hardly at all.)
        // Its color follows the beater too: white noise is a stick on
        // a rim; a felt beater's landing has little above 1 kHz.
        if self.click_env > 1e-4 {
            let hard = self.p.hardness.clamp(0.0, 1.0);
            let ms = 0.5 + 2.5 * (1.0 - hard);
            self.click_env *= (-1.0 / (ms * 0.001 * SR_INT)).exp();
            let cutoff = 600.0 * (10.0f32).powf(hard);
            let a = (std::f32::consts::TAU * cutoff / SR_INT).min(1.0);
            self.click_lp += a * (rng.next() - self.click_lp);
            // Lowpassed noise is quieter; lift it back in proportion.
            out += self.click_lp * self.click_env * (1.0 / a.sqrt()).min(2.5);
        }
        out
    }

    pub fn tick(&mut self, rng: &mut Rng) -> f32 {
        if self.quiet > (2.0 * SR) as u32 {
            return 0.0; // asleep: rang silent long ago
        }
        if self.phase == 0 {
            self.out_a = self.out_b;
            self.out_b = self.step(rng);
            if self.energy < 1e-9 && self.out_b.abs() < 1e-5 {
                self.quiet += DECIM as u32;
            } else {
                self.quiet = 0;
            }
        }
        let t = self.phase as f32 / DECIM as f32;
        self.phase = (self.phase + 1) % DECIM;
        let mut y = self.out_a + (self.out_b - self.out_a) * t;
        if self.p.drive > 0.0 {
            y = (self.p.drive * y).tanh() / self.p.drive.tanh();
        }
        y * self.p.level
    }
}

/// The mesh kit's defaults, calibrated against recorded hits
/// (samples/, examples/analyze.rs) as far as a single head goes:
/// fundamentals, decays, bends, (1,1) levels. Two things the
/// recordings show that the defaults only gesture at: real drums have
/// two heads, so their spectra carry a second, interleaved mode
/// series (the `air`/resonant-head pair supplies the strongest one),
/// and tom overtones ring about as long as the fundamental (overtone
/// damp near 0), which is *not* what the hand-tuned modal kit does.
pub fn default_kit() -> Vec<(&'static str, MeshParams)> {
    let base = MeshParams {
        freq: 110.0,
        decay: 0.4,
        hf_damp: 0.05,
        tension: 10.0,
        strike_pos: 0.3,
        rattle: 0.0,
        click: 0.0,
        drive: 0.0,
        air: 0.0,
        reso_freq: 100.0,
        reso_decay: 0.4,
        hardness: 0.7,
        mallet: 1.5,
        level: 0.6,
    };
    vec![
        // 18" jazz kick: 46 Hz, dead in 70 ms, felt beater, a
        // shell's worth of air; the coupled pair carries what tail
        // there is.
        (
            "kick",
            MeshParams {
                freq: 50.0,
                decay: 0.1,
                hf_damp: 0.3,
                tension: 25.0,
                strike_pos: 0.12,
                click: 0.5,
                drive: 1.5,
                air: 0.3,
                reso_freq: 42.0,
                reso_decay: 0.15,
                hardness: 0.05,
                mallet: 3.0,
                level: 2.0,
                ..base
            },
        ),
        // Dry snare: the head is gone in 40 ms and bends a quarter
        // tone… a fifth; the wires are the note.
        (
            "snare",
            MeshParams {
                freq: 170.0,
                decay: 0.04,
                hf_damp: 0.2,
                tension: 24.0,
                strike_pos: 0.4,
                rattle: 0.8,
                click: 0.2,
                hardness: 0.9,
                mallet: 1.2,
                ..base
            },
        ),
        // 12" rack tom: 114 Hz, τ 0.32 s, +12% bend, a second head's
        // mode 10–15% above the fundamental. (A standard kit is two
        // rack toms and a floor tom; jazz kits often one of each,
        // rock kits sprawl to four or more.)
        (
            "tom hi",
            MeshParams {
                freq: 114.0,
                decay: 0.22,
                tension: 8.0,
                air: 0.06,
                reso_freq: 130.0,
                reso_decay: 0.2,
                hardness: 0.9,
                click: 0.15,
                strike_pos: 0.4,
                ..base
            },
        ),
        // 14" floor tom, between the two recorded ones: 95 Hz.
        (
            "tom",
            MeshParams {
                freq: 95.0,
                decay: 0.4,
                tension: 10.0,
                air: 0.1,
                reso_freq: 116.0,
                reso_decay: 0.22,
                ..base
            },
        ),
        // 16" floor tom: 80 Hz, τ 0.5 s, +20% bend, (1,1) at −15 dB.
        (
            "tom lo",
            MeshParams {
                freq: 80.0,
                decay: 0.3,
                tension: 12.0,
                air: 0.12,
                reso_freq: 100.0,
                reso_decay: 0.25,
                hardness: 0.9,
                click: 0.15,
                strike_pos: 0.4,
                ..base
            },
        ),
    ]
}

/// A kit of meshes.
pub struct Kit {
    pads: Vec<Mesh>,
}

impl Kit {
    pub fn new(params: Vec<MeshParams>) -> Self {
        Kit {
            pads: params.into_iter().map(Mesh::new).collect(),
        }
    }
    pub fn set_params(&mut self, i: usize, p: MeshParams) {
        if let Some(m) = self.pads.get_mut(i) {
            m.set_params(p);
        }
    }
    pub fn strike(&mut self, i: usize, strength: f32) {
        if let Some(m) = self.pads.get_mut(i) {
            m.strike(strength);
        }
    }
    pub fn surface(&self, i: usize) -> Option<&[f32]> {
        self.pads.get(i).map(|m| m.surface())
    }
    pub fn tick(&mut self, rng: &mut Rng) -> f32 {
        self.pads.iter_mut().map(|m| m.tick(rng)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn tom() -> MeshParams {
        MeshParams {
            freq: 110.0,
            decay: 0.6,
            hf_damp: 0.3,
            tension: 0.0,
            strike_pos: 0.5,
            rattle: 0.0,
            click: 0.0,
            drive: 0.0,
            air: 0.0,
            reso_freq: 50.0,
            reso_decay: 0.5,
            hardness: 0.7,
            mallet: 1.5,
            level: 1.0,
        }
    }

    fn render(p: MeshParams, strength: f32, secs: f32) -> Vec<f32> {
        let mut m = Mesh::new(p);
        let mut rng = Rng(3);
        m.strike(strength);
        (0..(secs * SR) as usize)
            .map(|_| m.tick(&mut rng))
            .collect()
    }

    fn peaks(buf: &[f32], lo: f32, hi: f32, step: f32) -> Vec<(f32, f32)> {
        let bands: Vec<(f32, f32)> = (0..)
            .map(|k| lo + k as f32 * step)
            .take_while(|&f| f <= hi)
            .map(|f| {
                let w = TAU * f / SR;
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for (n, s) in buf.iter().enumerate() {
                    re += s * (w * n as f32).cos();
                    im += s * (w * n as f32).sin();
                }
                (f, (re * re + im * im).sqrt())
            })
            .collect();
        let mut out = Vec::new();
        for i in 1..bands.len() - 1 {
            if bands[i].1 > bands[i - 1].1 && bands[i].1 >= bands[i + 1].1 {
                out.push(bands[i]);
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }

    #[test]
    fn disc_rings_at_its_fundamental_and_decays() {
        let out = render(tom(), 1.0, 1.5);
        assert!(out.iter().all(|s| s.is_finite()));
        let pk = peaks(&out[..44100], 60.0, 200.0, 2.0);
        let f1 = pk[0].0;
        assert!(
            (f1 / 110.0 - 1.0).abs() < 0.08,
            "fundamental {f1} Hz (wanted 110)"
        );
        let early: f32 = out[..4410].iter().map(|s| s * s).sum::<f32>();
        let late: f32 = out[52920..57330].iter().map(|s| s * s).sum::<f32>();
        assert!(
            early > 1e-3 && late < early * 0.1,
            "no decay: {early} -> {late}"
        );
    }

    #[test]
    fn bessel_ratios_emerge() {
        // Nobody typed 1.59 in: an off-center strike must show the
        // (1,1) mode near 1.59× the fundamental.
        let out = render(tom(), 1.0, 1.0);
        let pk = peaks(&out, 60.0, 400.0, 2.0);
        let f1 = 110.0;
        let has = |ratio: f32| {
            pk.iter()
                .take(6)
                .any(|(f, _)| ((f / f1) / ratio - 1.0).abs() < 0.08)
        };
        assert!(
            has(1.59),
            "no (1,1) mode near 1.59×: {:?}",
            &pk[..6.min(pk.len())]
        );
    }

    #[test]
    fn hard_hits_die_out() {
        // The tension modulation must not pump the head: after a hard
        // hit, or a burst of them, the tail must fall at the same rate
        // as with the tension off. (The first version rang forever at
        // the tension cap.) Compared over the second half of the tail —
        // the hits themselves land differently on a stiffened head, so
        // absolute levels are not the test.
        let tail = |tension: f32, hits: usize| {
            let mut m = Mesh::new(MeshParams { tension, ..tom() });
            let mut rng = Rng(3);
            let mut out = Vec::new();
            for h in 0..hits {
                m.strike(1.6);
                let n = if h + 1 == hits {
                    2 * SR as usize
                } else {
                    (0.12 * SR) as usize
                };
                out.extend((0..n).map(|_| m.tick(&mut rng)));
            }
            let n = out.len();
            let rms = |a: usize, b: usize| {
                (out[a..b].iter().map(|s| s * s).sum::<f32>() / (b - a) as f32).sqrt()
            };
            let mid = rms(n - SR as usize, n - SR as usize + 2205);
            let late = rms(n - 2205, n);
            (mid, late)
        };
        for hits in [1, 8] {
            let (mid, late) = tail(6.0, hits);
            let (mid0, late0) = tail(0.0, hits);
            let (drop, drop0) = (late / mid, late0 / mid0);
            assert!(
                drop < 0.5 && drop < drop0 * 1.3,
                "{hits} hard hit(s): tail {mid:.4} -> {late:.4} over a second \
                 (tension off: {mid0:.4} -> {late0:.4})"
            );
        }
    }

    /// Spectral centroid of the first 50 ms, 50 Hz..6 kHz.
    fn brightness(out: &[f32]) -> f32 {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let mut f = 50.0;
        while f < 6000.0 {
            let w = TAU * f / SR;
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, s) in out[..2205].iter().enumerate() {
                re += s * (w * n as f32).cos();
                im += s * (w * n as f32).sin();
            }
            let mag = (re * re + im * im).sqrt();
            num += f * mag;
            den += mag;
            f *= 1.05;
        }
        num / den
    }

    #[test]
    fn stick_contact_shapes_the_attack() {
        // Nothing here is a filter: a harder tip, or a harder hit,
        // leaves the stick on the head for less time and the attack
        // comes out brighter — the collision decides the spectrum.
        let felt = render(
            MeshParams {
                hardness: 0.0,
                ..tom()
            },
            1.0,
            0.3,
        );
        let wood = render(
            MeshParams {
                hardness: 1.0,
                ..tom()
            },
            1.0,
            0.3,
        );
        let (bf, bw) = (brightness(&felt), brightness(&wood));
        assert!(bw > bf * 1.5, "felt {bf:.0} Hz vs wood {bw:.0} Hz");
        let soft = render(tom(), 0.4, 0.3);
        let hard = render(tom(), 1.6, 0.3);
        let (bs, bh) = (brightness(&soft), brightness(&hard));
        assert!(bh > bs * 1.1, "soft hit {bs:.0} Hz vs hard hit {bh:.0} Hz");
        let mut m = Mesh::new(tom());
        let mut rng = Rng(3);
        m.strike(1.0);
        for _ in 0..4410 {
            m.tick(&mut rng);
        }
        let ms = m.contact_steps() as f32 * DECIM as f32 / SR * 1000.0;
        assert!((1.0..6.0).contains(&ms), "stick on head {ms:.1} ms");
    }

    #[test]
    fn resonant_head_carries_the_tail() {
        // With the air cavity on, the batter's energy moves into the
        // resonant head: the tail is longer and the resonant head's
        // pitch is in it. And it all still dies away.
        let kick = MeshParams {
            freq: 60.0,
            decay: 0.15,
            hf_damp: 0.5,
            strike_pos: 0.05,
            hardness: 0.1,
            mallet: 3.0,
            ..tom()
        };
        let open = render(kick, 1.0, 1.5);
        let shelled = render(
            MeshParams {
                air: 0.6,
                reso_freq: 48.0,
                reso_decay: 0.6,
                ..kick
            },
            1.0,
            1.5,
        );
        assert!(shelled.iter().all(|s| s.is_finite()));
        let rms = |b: &[f32]| (b.iter().map(|s| s * s).sum::<f32>() / b.len() as f32).sqrt();
        let (o_early, o_late) = (rms(&open[..4410]), rms(&open[22050..26460]));
        let (s_early, s_late) = (rms(&shelled[..4410]), rms(&shelled[22050..26460]));
        assert!(
            s_late / s_early > 2.0 * o_late / o_early,
            "tail not longer: open {o_early:.3}->{o_late:.4}, shelled {s_early:.3}->{s_late:.4}"
        );
        let end = rms(&shelled[61740..66150]);
        assert!(end < s_early * 0.05, "shelled kick still ringing: {end:.4}");
        // Strongly coupled, the two heads' modes merge into a pair
        // around the tuning; what must hold is that the ring stays
        // deep — the pair, not some upper mode, carries the tail.
        let pk = peaks(&shelled[4410..48510], 30.0, 200.0, 1.0);
        assert!(
            (40.0..75.0).contains(&pk[0].0),
            "tail not deep: strongest ring at {} Hz",
            pk[0].0
        );
    }

    #[test]
    fn tension_makes_hard_hits_sharp() {
        let p = MeshParams {
            tension: 12.0,
            ..tom()
        };
        let onset = |strength: f32| {
            let out = render(p, strength, 0.5);
            peaks(&out[..8820], 80.0, 140.0, 2.0)[0].0
        };
        let soft = onset(0.2);
        let hard = onset(1.5);
        assert!(hard > soft * 1.02, "no glide: soft {soft} hard {hard}");
    }
}
