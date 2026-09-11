//! Model 5: the Kelly-Lochbaum vocal tract — the voice as a tube, not
//! a filter bank.
//!
//! The tract is ~44 short cylindrical segments of varying cross-section,
//! run as a two-rail digital waveguide: right- and left-going pressure
//! waves with a scattering junction wherever the area changes
//! (reflection k = (A0 - A1)/(A0 + A1)) — the bowed string's
//! architecture with forty-four junctions instead of two. The glottis
//! reflects at one end (and injects the voice source); the lips
//! partially reflect and partially radiate at the other, and what
//! radiates is what you hear.
//!
//! Nothing here knows what a formant is. Vowels are *emergent*: shape
//! the tube like a mouth saying /a/ and the resonances of that shape
//! are the formants of /a/, moving together along physical paths as the
//! shape deforms — which is precisely what independent formant sliders
//! (voice.rs, mouth.rs) can never do. Constrict the tube far enough and
//! injected turbulence becomes a fricative: consonants from geometry.
//!
//! The tract ladder runs two substeps per output sample, halving the
//! per-section acoustic length so 44 sections come out at ~17 cm — an
//! adult vocal tract — at 44.1 kHz.

use crate::util::{AtomicF32, Rng};
use crate::voice::Glottis;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// Number of tube sections.
pub const N: usize = 44;

/// Nasal tract sections (~12 cm of nose at the same section length).
pub const NOSE: usize = 28;
/// Oral section index where the velum port sits.
const VELUM_AT: usize = 17;
/// Fully open velum port diameter.
const VELUM_D: f32 = 0.7;
/// The nose is soft-walled (mucosa): slightly lossier than the mouth.
const NOSE_DAMP: f32 = 0.997;
const NOSE_REFL: f32 = -0.78;

/// The nose's fixed shape: opening from the velum port, widest
/// mid-cavity, tapering to the nostrils.
fn nose_diameters() -> [f32; NOSE] {
    let mut d = [0.0f32; NOSE];
    for (i, di) in d.iter_mut().enumerate() {
        let x = 2.0 * i as f32 / (NOSE - 1) as f32;
        *di = if x < 1.0 {
            0.4 + 1.1 * x
        } else {
            0.5 + 1.0 * (2.0 - x)
        }
        .min(1.4);
    }
    d
}

/// Glottal reflection with folds nearly closed (voiced posture)...
const GLOTTAL_REFL: f32 = 0.75;
/// ...and with the glottis wide open (whisper/aspiration): the tract
/// couples into the subglottal airways, reflections weaken, and the
/// formants broaden — the diffuse quality of whispered speech. A fixed
/// closed-glottis boundary makes every /h/ ring like noise through a
/// resonant pipe.
const GLOTTAL_REFL_OPEN: f32 = 0.3;
const LIP_REFL: f32 = -0.85;
/// Per-section wave damping — the tract's wall losses.
const DAMP: f32 = 0.999;
/// Coefficients refresh cadence in samples (diameters smooth per sample;
/// the divisions live here).
const REFRESH: u32 = 16;

pub struct Ctl {
    pub freq: AtomicF32,
    /// Where along the mouth the tongue rises, 0 (back/pharynx) to
    /// 1 (front/teeth).
    pub tongue_pos: AtomicF32,
    /// How far the tongue closes the tube, 0 (open) to 1 (toward
    /// closure). The pad clamps its input below the frication zone;
    /// deliberate fricatives and stops come from the speech sequencer,
    /// which may push past 1.0.
    pub constrict: AtomicF32,
    /// Lip opening, 0 (rounded/closed) to 1 (spread).
    pub lips: AtomicF32,
    /// Velum opening, 0 (sealed nose) to 1 (open port). Nonzero during
    /// vowels = nasalized vowels; the sequencer opens it for m/n/ng.
    pub velum: AtomicF32,
    /// Set by the audio thread while a speak::Utterance is running, so
    /// the UI can route keys to the talking voice.
    pub speaking: AtomicBool,
    pub breath: AtomicF32,
    pub vibrato: AtomicF32,
    pub level: AtomicF32,
    pub gate: AtomicBool,
}

impl Ctl {
    pub fn new() -> Self {
        Ctl {
            freq: AtomicF32::new(120.0),
            tongue_pos: AtomicF32::new(0.5),
            constrict: AtomicF32::new(0.3),
            lips: AtomicF32::new(0.8),
            velum: AtomicF32::new(0.0),
            speaking: AtomicBool::new(false),
            breath: AtomicF32::new(0.06),
            vibrato: AtomicF32::new(0.01),
            level: AtomicF32::new(0.8),
            gate: AtomicBool::new(false),
        }
    }
}

impl Default for Ctl {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct Params {
    pub freq: f32,
    pub tongue_pos: f32,
    pub constrict: f32,
    pub lips: f32,
    pub velum: f32,
    /// Tongue-tip raising, 0 (down) to 1 (sealed at the ridge).
    pub tip: f32,
    /// Glottal (voiced) source amount. The pad derives it from breath;
    /// the speech sequencer sets it independently — a "t" closure is
    /// silent, not breathy.
    pub voiced: f32,
    pub breath: f32,
    /// Is air being pushed through the tract? Turbulence needs flow AND
    /// an opening: a stop closure has neither, however tight it is.
    pub flow: f32,
    pub vibrato: f32,
    pub level: f32,
    pub gate: bool,
}

impl Params {
    pub fn read(ctl: &Ctl) -> Self {
        Params {
            freq: ctl.freq.get(),
            tongue_pos: ctl.tongue_pos.get().clamp(0.0, 1.0),
            constrict: ctl.constrict.get().clamp(0.0, 1.0),
            lips: ctl.lips.get().clamp(0.0, 1.0),
            velum: ctl.velum.get().clamp(0.0, 1.0),
            tip: 0.0,
            voiced: 1.0 - ctl.breath.get(),
            breath: ctl.breath.get(),
            flow: 1.0,
            vibrato: ctl.vibrato.get(),
            level: ctl.level.get(),
            gate: ctl.gate.load(Relaxed),
        }
    }
}

/// Tongue-tip hump: fixed at the alveolar ridge, narrow, raised
/// independently of the body — /n t d l/ flick it while the body holds
/// the vowel (a single-hump tongue must haul the whole body up and
/// back through /j/-space instead, which is audible).
const TIP_AT: f32 = 0.78;
const TIP_W: f32 = 0.07;

/// The articulation → tube-shape map, shared with the UI so the drawn
/// tract is exactly what the audio thread scatters through. Diameters
/// in arbitrary units (~cm-ish); areas are their squares.
pub fn diameters(tongue_pos: f32, constrict: f32, lips: f32, tip: f32) -> [f32; N] {
    let mut d = [0.0f32; N];
    for (i, di) in d.iter_mut().enumerate() {
        let x = i as f32 / (N - 1) as f32;
        // Rest shape: narrow above the glottis, opening through the
        // pharynx into a wide mouth.
        // Seams matched at both boundaries: an unmatched step is a
        // permanent spurious scattering junction mid-tract.
        *di = if x < 0.15 {
            0.4 + 2.0 * x
        } else if x < 0.47 {
            0.7 + (x - 0.15) * 2.5
        } else {
            1.5
        };
    }
    // The tongue: a raised-cosine hump narrowing the tube, its center
    // sweeping the back-to-front span of the oral cavity.
    let center = 0.30 + 0.55 * tongue_pos;
    let width = 0.16;
    for (i, di) in d.iter_mut().enumerate() {
        let x = i as f32 / (N - 1) as f32;
        let t = (x - center) / width;
        if t.abs() < 1.0 {
            let bump = 0.5 * (1.0 + (PI * t).cos());
            // Constriction narrows toward closure; the sequencer may
            // push constrict past 1.0 for true stops, floored at 0.03.
            *di = (*di - constrict * bump * (*di - 0.12)).max(0.03);
        }
    }
    // The tongue tip: raised in place, never swept. It is an
    // independent APERTURE, composed with min() — subtracting from the
    // body's diameter would stack when a front vowel's constriction
    // sits under the ridge (/i/ does), over-pinching into frication.
    if tip > 0.0 {
        for (i, di) in d.iter_mut().enumerate() {
            let x = i as f32 / (N - 1) as f32;
            let t = (x - TIP_AT) / TIP_W;
            if t.abs() < 1.0 {
                let bump = 0.5 * (1.0 + (PI * t).cos());
                let ceiling = (0.02 + 1.48 * (1.0 - tip * bump)).max(0.03);
                *di = di.min(ceiling);
            }
        }
    }
    // The lips: the last few sections. Floor low enough that lips 0 is
    // a genuine closure — the sequencer's /p/ and /b/ need it.
    for di in d.iter_mut().skip(N - 4) {
        *di = di.min(0.05 + 1.55 * lips);
    }
    d
}

/// Calibrated articulations: (name, tongue_pos, constrict, lips),
/// found by scanning the articulation space with examples/tractprobe.rs
/// and picking the shapes whose *emergent* resonances land the classic
/// vowel formants (verified by the tests below).
pub const VOWELS: [(&str, f32, f32, f32); 5] = [
    ("ah", 0.05, 0.80, 1.0),
    ("ee", 0.80, 0.85, 0.91),
    ("oo", 0.40, 0.85, 0.28),
    ("eh", 0.60, 0.60, 1.0),
    ("oh", 0.20, 0.80, 0.27),
];

/// Articulation and source-mix smoothing: a touch faster than the
/// knobs — tongues move quickly.
const ARTIC: f32 = 0.004;

pub struct Tract {
    // Tube state.
    diam: [f32; N],
    // refl[i]: junction between sections i-1 and i. Targets come from
    // the shape refresh; the live values glide per sample — stepped
    // coefficients in an energized tube click, worst at a nasal's
    // closure where near-zero areas make the formula hypersensitive.
    refl: [f32; N],
    refl_t: [f32; N],
    right: [f32; N],
    left: [f32; N],
    // The nose: a fixed side branch off the velum junction.
    nose_r: [f32; NOSE],
    nose_l: [f32; NOSE],
    nose_refl: [f32; NOSE],
    velum_s: f32,
    // Three-port scattering coefficients at the velum junction. The
    // refresh computes targets; the live values glide there per sample
    // (a stepped junction coefficient in an energized tube clicks).
    // With the port closed they collapse algebraically to the plain
    // two-port junction.
    vr_left: f32,
    vr_right: f32,
    vr_nose: f32,
    vr_left_t: f32,
    vr_right_t: f32,
    vr_nose_t: f32,
    refresh: u32,
    min_d: f32,
    min_i: usize,
    /// The shared humanized source (voice::Glottis) — one larynx
    /// implementation for both streaming engines.
    g: Glottis,
    // Smoothed source-mix parameters: the sequencer steps them per
    // segment, and an unsmoothed step through the radiation
    // differentiator is an audible pop.
    voiced_s: f32,
    breath_s: f32,
    flow_s: f32,
    turb_prev: f32,
    turb_lp: f32,
    asp_lp: f32,
    asp_lp2: f32,
    /// Slow random flutter on the breath — turbulent flow is unsteady,
    /// and rock-steady noise reads as a gas cylinder, not lungs.
    flutter: f32,
    rad_prev: f32,
    rad_lp: f32,
    dcb_x: f32,
    dcb_y: f32,
}

impl Tract {
    pub fn new() -> Self {
        Tract {
            diam: diameters(0.5, 0.3, 0.8, 0.0),
            refl: [0.0; N],
            refl_t: [0.0; N],
            right: [0.0; N],
            left: [0.0; N],
            nose_r: [0.0; NOSE],
            nose_l: [0.0; NOSE],
            nose_refl: {
                let d = nose_diameters();
                let mut r = [0.0f32; NOSE];
                for i in 1..NOSE {
                    let a0 = d[i - 1] * d[i - 1];
                    let a1 = d[i] * d[i];
                    r[i] = (a0 - a1) / (a0 + a1).max(1e-6);
                }
                r
            },
            velum_s: 0.0,
            vr_left: 0.0,
            vr_right: 0.0,
            vr_nose: -1.0,
            vr_left_t: 0.0,
            vr_right_t: 0.0,
            vr_nose_t: -1.0,
            refresh: 0,
            min_d: 1.5,
            min_i: N / 2,
            g: Glottis::new(120.0),
            voiced_s: 0.0,
            breath_s: 0.0,
            flow_s: 1.0,
            turb_prev: 0.0,
            turb_lp: 0.0,
            asp_lp: 0.0,
            asp_lp2: 0.0,
            flutter: 0.0,
            rad_prev: 0.0,
            rad_lp: 0.0,
            dcb_x: 0.0,
            dcb_y: 0.0,
        }
    }

    fn refresh_shape(&mut self) {
        let mut min_d = f32::MAX;
        let mut min_i = N / 2;
        for i in 1..N {
            let a0 = self.diam[i - 1] * self.diam[i - 1];
            let a1 = self.diam[i] * self.diam[i];
            self.refl_t[i] = (a0 - a1) / (a0 + a1).max(1e-6);
            // Constriction search in the oral region only.
            if i >= 8 && self.diam[i] < min_d {
                min_d = self.diam[i];
                min_i = i;
            }
        }
        self.min_d = min_d;
        self.min_i = min_i;
        // Velum junction: three-port scattering among the oral sections
        // either side and the nose inlet, whose area IS the velum
        // opening.
        let a_l = self.diam[VELUM_AT - 1] * self.diam[VELUM_AT - 1];
        let a_r = self.diam[VELUM_AT] * self.diam[VELUM_AT];
        let vd = self.velum_s * VELUM_D;
        let a_n = vd * vd;
        let sum = (a_l + a_r + a_n).max(1e-6);
        self.vr_left_t = (2.0 * a_l - sum) / sum;
        self.vr_right_t = (2.0 * a_r - sum) / sum;
        self.vr_nose_t = (2.0 * a_n - sum) / sum;
    }

    pub fn tick(&mut self, p: &Params, rng: &mut Rng) -> f32 {
        let g = self.g.tick(p.freq, p.vibrato, p.gate, rng);
        if self.g.amp < 1e-4 && !p.gate {
            return 0.0;
        }
        let amp = self.g.amp;

        // Articulation: smooth every diameter toward the target shape —
        // but only while phonating. With the gate off the mouth holds
        // its posture while the level releases: relaxing with the
        // source still fading coins a phantom syllable ("...buh").
        if p.gate {
            let tgt = diameters(p.tongue_pos, p.constrict, p.lips, p.tip);
            for (d, t) in self.diam.iter_mut().zip(tgt) {
                *d += ARTIC * (t - *d);
            }
            // Velum and source-mix hold with the articulation when the
            // gate is off: the release is "everything holds, the
            // breath runs out". Letting them drift to the pad's
            // settings mid-fade reseals the nose and half re-lights
            // the voice inside a dying note (final nasals popped).
            self.velum_s += ARTIC * (p.velum - self.velum_s);
            self.voiced_s += ARTIC * (p.voiced - self.voiced_s);
            self.breath_s += ARTIC * (p.breath - self.breath_s);
            self.flow_s += ARTIC * (p.flow - self.flow_s);
        }
        if self.refresh == 0 {
            self.refresh = REFRESH;
            self.refresh_shape();
        }
        self.refresh -= 1;

        // Aspiration: dark (lowpassed) noise, pulsed by the glottal
        // cycle only to the extent the folds are actually vibrating —
        // an unvoiced /h/ is a continuous whisper, not pitch-chopped
        // hiss. (It is NOT steady in level: `flutter` below wobbles it
        // slowly, because rock-steady breath reads as a gas cylinder.)
        let v01 = self.voiced_s.clamp(0.0, 1.0);
        let pulse = 0.3 + 0.7 * (g * v01 + 0.5 * (1.0 - v01));
        // Corner ~800 Hz: glottal jet noise concentrates low and falls
        // steeply — a bright source hisses through any open mouth
        // (rounded vowels self-filter at the lips; open ones can't).
        self.asp_lp += 0.11 * (rng.next() - self.asp_lp);
        self.asp_lp2 += 0.11 * (self.asp_lp - self.asp_lp2);
        self.flutter += 0.002 * (rng.next() - self.flutter);
        let unsteady = (1.0 + 30.0 * self.flutter).clamp(0.25, 1.75);
        let src = (g * self.voiced_s + self.asp_lp2 * 4.0 * self.breath_s * pulse * unsteady) * amp;

        // Glide all junction coefficients toward their targets.
        for i in 1..N {
            self.refl[i] += 0.01 * (self.refl_t[i] - self.refl[i]);
        }
        self.vr_left += 0.01 * (self.vr_left_t - self.vr_left);
        self.vr_right += 0.01 * (self.vr_right_t - self.vr_right);
        self.vr_nose += 0.01 * (self.vr_nose_t - self.vr_nose);

        // Turbulence: air forced through a tight constriction goes
        // noisy — the fricative mechanism, from geometry alone.
        let mut turb = 0.0;
        // Onset at 0.30: every vowel's constriction must sit OUTSIDE
        // the frication zone (tightest is /a/ at ~0.32 after its
        // constrict was eased for exactly this reason) — vowels carry
        // no intrinsic hiss, and the lip-radiation tilt mercilessly
        // amplifies any that leaks.
        if self.min_d < 0.30 {
            // First-differenced noise: turbulence is generated bright —
            // raw white noise reads as a leaky pipe, not an /s/. It
            // scales with the squared narrowness, is gated by whether
            // any opening remains (a stop closure is silent however
            // tight), and by whether air is flowing at all.
            let x = (0.30 - self.min_d) / 0.30;
            let open = ((self.min_d - 0.05) / 0.08).clamp(0.0, 1.0);
            let n = rng.next();
            let bright = n - self.turb_prev;
            self.turb_prev = n;
            // Band-shape it: differencing alone rises to the Nyquist —
            // gas-jet territory. A lowpass after the difference centers
            // the noise around the real fricative band (~4-8 kHz).
            self.turb_lp += 0.55 * (bright - self.turb_lp);
            // The open velum diverts flow through the nose: a closure
            // pressing shut during a nasal passes through perfect
            // fricative geometry, but with almost no oral airflow to
            // go turbulent — without this factor every nasal press and
            // lift fires a frication tick (the "double click").
            // The shunt is low-impedance: above half-open, effectively
            // all flow goes nasal (a half-closed velum during a lift
            // otherwise lets the re-crossing constriction half-hiss).
            let oral_flow = self.flow_s * (1.0 - 2.0 * self.velum_s).clamp(0.0, 1.0);
            turb = self.turb_lp * 0.9 * x * x * open * oral_flow * amp;
        }

        // The ladder, two substeps per output sample.
        let mut out = 0.0;
        for _ in 0..2 {
            let mut jr = [0.0f32; N + 1];
            let mut jl = [0.0f32; N + 1];
            let mut nose_in = 0.0f32;
            let grefl = GLOTTAL_REFL_OPEN + (GLOTTAL_REFL - GLOTTAL_REFL_OPEN) * v01;
            jr[0] = self.left[0] * grefl + src;
            jl[N] = self.right[N - 1] * LIP_REFL;
            for i in 1..N {
                if i == VELUM_AT {
                    // Three-port junction: oral left, oral right, nose.
                    let (r_in, l_in, n_in) = (self.right[i - 1], self.left[i], self.nose_l[0]);
                    jr[i] = self.vr_right * l_in + (1.0 + self.vr_right) * (r_in + n_in);
                    jl[i] = self.vr_left * r_in + (1.0 + self.vr_left) * (l_in + n_in);
                    nose_in = self.vr_nose * n_in + (1.0 + self.vr_nose) * (l_in + r_in);
                } else {
                    let w = self.refl[i] * (self.right[i - 1] + self.left[i]);
                    jr[i] = self.right[i - 1] - w;
                    jl[i] = self.left[i] + w;
                }
            }
            for i in 0..N {
                self.right[i] = jr[i] * DAMP;
                self.left[i] = jl[i + 1] * DAMP;
            }
            // The nose ladder: fixed shape, nostril radiation. The
            // junction's fresh output enters here; interior junctions
            // read the previous state uniformly.
            let mut njr = [0.0f32; NOSE + 1];
            let mut njl = [0.0f32; NOSE + 1];
            njr[0] = nose_in;
            njl[NOSE] = self.nose_r[NOSE - 1] * NOSE_REFL;
            for i in 1..NOSE {
                let w = self.nose_refl[i] * (self.nose_r[i - 1] + self.nose_l[i]);
                njr[i] = self.nose_r[i - 1] - w;
                njl[i] = self.nose_l[i] + w;
            }
            for i in 0..NOSE {
                self.nose_r[i] = njr[i] * NOSE_DAMP;
                self.nose_l[i] = njl[i + 1] * NOSE_DAMP;
            }
            self.right[self.min_i] += turb;
            self.left[self.min_i] += turb;
            // Both apertures radiate: lips and nostrils.
            out += self.right[N - 1] + self.nose_r[NOSE - 1];
        }
        let out = out * 0.5;
        // Lip radiation: what a listener receives is (nearly) the
        // derivative of the mouth-opening flow — radiation efficiency
        // rises with frequency. Without this the raw fundamental booms
        // straight out and open vowels sound loud and robotic; with it
        // the spectrum tilts the way real mouths do. (Overall level is
        // restored by the ×3 at the end of the chain.)
        let y = out - 0.95 * self.rad_prev;
        self.rad_prev = out;
        // ...and the tilt saturates: radiation efficiency stops rising
        // a few kHz up, so a gentle rolloff (~8 kHz) follows the
        // differentiator — without it the top octave's noise floor is
        // hoisted +6 dB/oct forever.
        self.rad_lp += 0.68 * (y - self.rad_lp);
        // DC blocker: the leaky differentiator still passes the
        // Rosenberg pulse's DC (measured +0.028 on the strip — a thump
        // per note, and a shifted operating point for the master
        // limiter downstream).
        let hp = self.rad_lp - self.dcb_x + 0.995 * self.dcb_y;
        self.dcb_x = self.rad_lp;
        self.dcb_y = hp;
        hp * 3.0 * p.level
    }
}

impl Default for Tract {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::SR;
    use std::f32::consts::TAU;

    /// Formant estimate: drive the tube with breath noise (whitened
    /// against the known source and radiation curves), collect spectral
    /// peaks, then choose the (F1, F2) pair jointly — see the selection
    /// rules below.
    fn formants(tongue_pos: f32, constrict: f32, lips: f32) -> (f32, f32) {
        let mut tract = Tract::new();
        let mut rng = Rng(11);
        let p = Params {
            freq: 120.0,
            tongue_pos,
            constrict,
            lips,
            velum: 0.0,
            tip: 0.0,
            voiced: 0.0,
            breath: 1.0,
            flow: 1.0,
            vibrato: 0.0,
            level: 1.0,
            gate: true,
        };
        let n = (1.5 * SR) as usize;
        let buf: Vec<f32> = (0..n).map(|_| tract.tick(&p, &mut rng)).collect();
        let tail = &buf[n - SR as usize..];
        let bands: Vec<(f32, f32)> = (6..130)
            .map(|k| {
                let f = k as f32 * 25.0;
                let w = TAU * f / SR;
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for (i, s) in tail.iter().enumerate() {
                    re += s * (w * i as f32).cos();
                    im += s * (w * i as f32).sin();
                }
                // Whiten: divide out the deliberately dark breath drive
                // (aspiration lowpass) and the lip-radiation tilt, so
                // the tube's resonances are judged on flat excitation.
                let a = 0.89f32;
                let hlp = (0.11 * 0.11) / (1.0 - 2.0 * a * w.cos() + a * a);
                let b = 0.32f32;
                let hrad = (1.0 - 2.0 * 0.95 * w.cos() + 0.95 * 0.95).sqrt() * 0.68
                    / (1.0 - 2.0 * b * w.cos() + b * b).sqrt();
                (f, (re * re + im * im).sqrt() / (hlp * hrad))
            })
            .collect();
        let sm: Vec<f32> = (0..bands.len())
            .map(|i| {
                let lo = i.saturating_sub(1);
                let hi = (i + 2).min(bands.len());
                bands[lo..hi].iter().map(|b| b.1).sum::<f32>() / (hi - lo) as f32
            })
            .collect();
        let mut peaks = Vec::new();
        for i in 2..sm.len() - 2 {
            if sm[i] > sm[i - 1] && sm[i] >= sm[i + 1] && sm[i] > sm[i - 2] && sm[i] > sm[i + 2] {
                peaks.push((bands[i].0, sm[i]));
            }
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        // Keep a wide net: whitening boosts the 3 kHz region, and a
        // top-3 cut can push the true F2 out before the rules below
        // ever see it.
        peaks.truncate(5);
        peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Joint selection, like a real formant tracker: pick the
        // (F1, F2) pair maximizing combined magnitude. Single-peak
        // rules fail here — weak spurs flank F1 (/a/), and /u/'s F2
        // is its strongest peak while sitting in F1's range.
        let mut best = (0.0f32, 0.0f32, f32::MIN);
        for &(f1, m1) in peaks.iter().filter(|(f, _)| (150.0..=950.0).contains(f)) {
            for &(f2, m2) in &peaks {
                // The 1/f1 weight is the tracker's prior: F1 is by
                // definition the FIRST resonance, so between comparable
                // candidates the lower one wins (/u/'s diffuse low
                // cluster otherwise flips with every drive change).
                let score = m1 * m2 / f1;
                if f2 > f1 + 150.0 && f2 <= 2600.0 && score > best.2 {
                    best = (f1, f2, score);
                }
            }
        }
        (best.0, best.1)
    }

    fn vowel(name: &str) -> (f32, f32) {
        let (_, tp, con, lips) = VOWELS.iter().find(|v| v.0 == name).unwrap();
        formants(*tp, *con, *lips)
    }

    #[test]
    fn vowels_emerge_from_tube_shapes() {
        // Nothing in tract.rs mentions formants; these resonances are
        // consequences of geometry. Targets are classic male values,
        // generously toleranced.
        let (f1, f2) = vowel("ah");
        assert!((600.0..=950.0).contains(&f1), "ah F1 {f1}");
        assert!((950.0..=1350.0).contains(&f2), "ah F2 {f2}");
        let (f1, f2) = vowel("ee");
        assert!((250.0..=450.0).contains(&f1), "ee F1 {f1}");
        // The probe drives an unvoiced (open-glottis) tract, and
        // whispered formants genuinely sit higher than voiced ones —
        // /i/'s F2 measures ~2500 whispered vs ~2050 voiced.
        assert!((1800.0..=2700.0).contains(&f2), "ee F2 {f2}");
        let (f1, f2) = vowel("oo");
        assert!((200.0..=400.0).contains(&f1), "oo F1 {f1}");
        assert!((650.0..=1000.0).contains(&f2), "oo F2 {f2}");
    }

    #[test]
    fn gate_sustains_then_releases_and_stays_finite() {
        let mut tract = Tract::new();
        let mut rng = Rng(11);
        let (_, tp, con, lips) = VOWELS[0];
        let on = Params {
            freq: 120.0,
            tongue_pos: tp,
            constrict: con,
            lips,
            velum: 0.0,
            tip: 0.0,
            voiced: 0.94,
            breath: 0.06,
            flow: 1.0,
            vibrato: 0.01,
            level: 0.8,
            gate: true,
        };
        let out: Vec<f32> = (0..44100).map(|_| tract.tick(&on, &mut rng)).collect();
        assert!(out.iter().all(|s| s.is_finite()));
        let held = (out[22050..].iter().map(|s| s * s).sum::<f32>() / 22050.0).sqrt();
        assert!(held > 0.01, "tract too quiet: {held}");
        let off = Params { gate: false, ..on };
        let tail: Vec<f32> = (0..22050).map(|_| tract.tick(&off, &mut rng)).collect();
        let released = (tail[11025..].iter().map(|s| s * s).sum::<f32>() / 11025.0).sqrt();
        assert!(released < held * 0.05, "tract kept sounding: {released}");
    }

    #[test]
    fn articulation_sweeps_stay_stable() {
        // Wild tongue gymnastics must never blow the ladder up.
        let mut tract = Tract::new();
        let mut rng = Rng(11);
        for i in 0..(2.0 * SR) as usize {
            let t = i as f32 / SR;
            let p = Params {
                freq: 140.0,
                tongue_pos: 0.5 + 0.5 * (t * 7.1).sin(),
                constrict: 0.5 + 0.5 * (t * 5.3).cos(),
                lips: 0.5 + 0.5 * (t * 3.7).sin(),
                velum: 0.5 + 0.5 * (t * 4.3).cos(),
                tip: 0.0,
                voiced: 0.8,
                breath: 0.2,
                flow: 1.0,
                vibrato: 0.01,
                level: 1.0,
                gate: true,
            };
            let y = tract.tick(&p, &mut rng);
            assert!(y.is_finite() && y.abs() < 25.0, "blew up at t={t}: {y}");
        }
    }

    #[test]
    fn velum_reroutes_sound_through_the_nose() {
        // Closed lips, voiced: with the velum open the hum radiates
        // from the nostrils (the /m/ murmur); sealed, the shut mouth is
        // near-silent.
        let hum = |velum: f32| {
            let mut tract = Tract::new();
            let mut rng = Rng(11);
            let p = Params {
                freq: 120.0,
                tongue_pos: 0.5,
                constrict: 0.35,
                lips: 0.0,
                velum,
                tip: 0.0,
                voiced: 1.0,
                breath: 0.02,
                flow: 1.0,
                vibrato: 0.0,
                level: 1.0,
                gate: true,
            };
            let out: Vec<f32> = (0..44100).map(|_| tract.tick(&p, &mut rng)).collect();
            assert!(out.iter().all(|s| s.is_finite()));
            (out[22050..].iter().map(|s| s * s).sum::<f32>() / 22050.0).sqrt()
        };
        let nasal = hum(1.0);
        let sealed = hum(0.0);
        assert!(
            nasal > 0.01 && nasal > sealed * 4.0,
            "no nasal murmur: open {nasal} sealed {sealed}"
        );
    }

    #[test]
    fn tight_constriction_makes_frication_noise() {
        // Voiced source, breath near zero: high-frequency energy above
        // 3 kHz must rise sharply when the tongue nearly closes —
        // turbulence from geometry, the consonant mechanism.
        let hf = |constrict: f32| {
            let mut tract = Tract::new();
            let mut rng = Rng(11);
            let p = Params {
                freq: 120.0,
                tongue_pos: 0.8,
                constrict,
                lips: 0.9,
                velum: 0.0,
                tip: 0.0,
                voiced: 0.98,
                breath: 0.02,
                flow: 1.0,
                vibrato: 0.0,
                level: 1.0,
                gate: true,
            };
            let n = 44100;
            let buf: Vec<f32> = (0..n).map(|_| tract.tick(&p, &mut rng)).collect();
            let tail = &buf[22050..];
            // Energy above ~3.5 kHz via first-difference emphasis.
            let mut acc = 0.0;
            for w in tail.windows(2) {
                let d = w[1] - w[0];
                acc += d * d;
            }
            (acc / tail.len() as f32).sqrt()
        };
        let open = hf(0.4);
        let tight = hf(0.97);
        assert!(
            tight > open * 3.0,
            "no frication: tight {tight} vs open {open}"
        );
    }
}
