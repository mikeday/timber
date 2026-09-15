//! The hi-hat: two small cymbals on a pedal. The instrument is the
//! *contact* between them. Open, both ring freely; closed, the top
//! plate is pressed onto the bottom one and the contact damps them
//! both — the tight "tss"; half open, the top plate chatters against
//! the bottom as it rings — the sizzle; and the pedal alone, dropping
//! the top plate onto the bottom, is the "chick" a drummer's foot
//! keeps time with.
//!
//! Two modal plates (cymbal.rs) sharing one mode set, meeting at a
//! ring of contact points near the edge through Hertz springs with
//! contact damping. The pedal moves the top plate as a whole; its
//! offset enters only the contact, so a fast close is an impact and a
//! hold is a pressure. Nothing here is a filter or a sample — the
//! choke is the physics of two pieces of bronze touching.

use std::sync::Arc;

use crate::cymbal::{Cymbal, CymbalParams, Modes};
use crate::util::Rng;

/// Contact points around the plates, at this fraction of the radius.
const CONTACTS: usize = 12;
const CONTACT_AT: f32 = 0.85;
const CONTACT_RAD: f32 = 2.0;
/// Hertz stiffness of the plate-on-plate contact, per-step² units.
const CONTACT_K: f32 = 4.0;
/// Contact damping: force per unit relative velocity while touching
/// (the impacts of chatter; explicit, so it must stay below ~1).
const CONTACT_DAMP: f32 = 0.6;
/// Per-point damping of each plate while pressed — pressed together,
/// the plates lock and ring as one (a pure spring transmits, it
/// doesn't absorb); what chokes a closed hat is the felt taking the
/// motion out of the pair. Applied implicitly, so any strength is
/// stable.
const PRESS_DAMP: f32 = 4.0;
/// The top plate as a rigid body: its mass in the contact's units,
/// and the pedal linkage as a spring and damper pulling it toward
/// where the foot wants it. The foot *throws* the plate: it arrives
/// at the bottom hat with momentum, and the contact absorbing that
/// momentum is the chick. (A prescribed position has infinite mass —
/// it stops where it is told, and the chick was nearly silent.)
const TOP_MASS: f32 = 30.0;
const PEDAL_K: f32 = 0.0012;
const PEDAL_DAMP: f32 = 0.25;
/// The foot's force with the pedal down: a *force*, not a position —
/// it accelerates the plate across the gap (the arrival is the
/// chick) and then holds it against the bending forces of a stick
/// hit (the choke).
const FOOT_FORCE: f32 = 0.01;
/// How far into the bottom plate the foot presses the top, as a
/// fraction of the gap.
const PRELOAD: f32 = 0.008;
/// The stand's felt and the bell: per-sample loss on the low modes,
/// tapering to nothing at STAND_CUT. A hat's lowest partials are its
/// weakest and shortest — with them ringing for seconds the open hat
/// was a hum at 600–800 Hz. Its voice is the top, which the plates'
/// own decay is left to carry.
const STAND_DAMP: f32 = 0.004;
const STAND_CUT: f32 = 1200.0;

#[derive(Clone, Copy)]
pub struct HiHatParams {
    /// The plates: small and thick, so stiff and high.
    pub plate: CymbalParams,
    /// Separation of the plates with the foot up, in displacement
    /// units; a small gap lets a ringing top plate chatter.
    pub gap: f32,
    /// How far the foot goes down with the pedal key, 0..1 of the
    /// gap: 1 is closed tight, less leaves it "half open".
    pub press: f32,
    pub level: f32,
}

pub struct HiHat {
    p: HiHatParams,
    top: Cymbal,
    bottom: Cymbal,
    /// Per-mode weights at each contact point, for each plate.
    w_top: Vec<Vec<f32>>,
    w_bot: Vec<Vec<f32>>,
    /// Foot 0 (up) .. 1 (down), and the top plate's rigid offset and
    /// velocity (the plate the foot is moving, with its own inertia).
    pedal_target: f32,
    z: f32,
    vz: f32,
    /// The foot is holding the plate down: the rigid body is a
    /// constraint now, not a mass on a spring (a mass on a spring
    /// pressed onto a Hertz contact bounces forever).
    latched: bool,
    /// Per-mode damping of each plate from the contacts touching this
    /// step (the diagonal of the point dampers projected on the modes;
    /// scratch, rebuilt per sample).
    d_top: Vec<f32>,
    d_bot: Vec<f32>,
}

impl HiHat {
    pub fn new(modes: Arc<Modes>, p: HiHatParams) -> Self {
        let (top_p, bot_p) = Self::plates(&p);
        let top = Cymbal::new(modes.clone(), top_p);
        let bottom = Cymbal::new(modes, bot_p);
        let mut h = HiHat {
            p,
            top,
            bottom,
            w_top: Vec::new(),
            w_bot: Vec::new(),
            pedal_target: 0.0,
            z: 0.0,
            vz: 0.0,
            latched: false,
            d_top: Vec::new(),
            d_bot: Vec::new(),
        };
        h.contacts();
        h
    }

    /// The pair: a bottom hat is heavier than its top (stiffer, a
    /// touch higher). Identical plates would also be a bug here: the
    /// contact pushes them apart symmetrically, and two identical
    /// discs heard at the same point cancel to the sample — the pedal
    /// alone made no sound at all.
    fn plates(p: &HiHatParams) -> (CymbalParams, CymbalParams) {
        let top = CymbalParams {
            level: 1.0,
            ..p.plate
        };
        let bottom = CymbalParams {
            stiffness: (p.plate.stiffness * 1.25).min(1.0),
            dome: p.plate.dome * 1.1,
            ..top
        };
        (top, bottom)
    }

    fn contacts(&mut self) {
        // Points around the edge: the disc helper places a point
        // along +x; the mode shapes are what they are on the grid, so
        // rotate by choosing points on the ring by hand.
        let (cx, cy) = self.top.point(CONTACT_AT);
        let center = self.top.center();
        let r = cx - center;
        self.w_top.clear();
        self.w_bot.clear();
        for k in 0..CONTACTS {
            let a = std::f32::consts::TAU * k as f32 / CONTACTS as f32;
            let pt = (center + r * a.cos(), cy + r * a.sin());
            self.w_top.push(self.top.weights_at(pt, CONTACT_RAD));
            self.w_bot.push(self.bottom.weights_at(pt, CONTACT_RAD));
        }
        let n = self.w_top[0].len();
        self.d_top = vec![0.0; n];
        self.d_bot = vec![0.0; n];
    }

    pub fn set_params(&mut self, p: HiHatParams) {
        self.p = p;
        let (top_p, bot_p) = Self::plates(&p);
        self.top.set_params(top_p);
        self.bottom.set_params(bot_p);
        self.contacts();
    }

    /// Stick on the top plate.
    pub fn strike(&mut self, strength: f32) {
        self.top.strike(strength);
        self.bottom.wake();
    }

    /// Foot down (true) or up.
    pub fn pedal(&mut self, down: bool) {
        self.pedal_target = if down {
            self.p.press.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.top.wake();
        self.bottom.wake();
    }

    /// Diagnostic: how many contact points are touching right now.
    pub fn touching(&self) -> usize {
        let offset = self.p.gap + self.z;
        (0..CONTACTS)
            .filter(|&k| {
                offset + self.top.displacement(&self.w_top[k])
                    - self.bottom.displacement(&self.w_bot[k])
                    < 0.0
            })
            .count()
    }

    pub fn tick(&mut self, rng: &mut Rng) -> f32 {
        // The stand: felt washers at the clamp take the lowest modes
        // out of both plates all the time (a hat never hums).
        self.top.stand_damp(STAND_DAMP, STAND_CUT);
        self.bottom.stand_damp(STAND_DAMP, STAND_CUT);
        // Where the foot wants the top plate: at the gap when up, a
        // hair into the bottom plate when down (the preload is tiny —
        // pressed-in plates are bent plates, and lifting the foot
        // released that bend as a ~200 Hz ring after every chick).
        let want = -self.pedal_target * self.p.gap * (1.0 + PRELOAD);
        // The linkage pulls the rigid plate toward it; the contact
        // pushes back below.
        let mut f_rigid =
            PEDAL_K * (want - self.z) - PEDAL_DAMP * self.vz - FOOT_FORCE * self.pedal_target;
        // Arrived under the foot: latch. Lifted: let go.
        if self.pedal_target > 0.0 && !self.latched && self.z <= want && self.vz <= 0.0 {
            self.latched = true;
        }
        if self.pedal_target == 0.0 {
            self.latched = false;
        }
        if self.latched {
            self.z = want;
            self.vz = 0.0;
        }
        let offset = self.p.gap + self.z;
        let v_rigid = self.vz;
        // Contacts: where the plates meet, a spring and a damper — and,
        // while the foot holds them there, the felt's damping on each
        // touching point (implicit; see cymbal.rs).
        let mut f_top = vec![0.0f32; self.w_top[0].len()];
        let mut f_bot = vec![0.0f32; self.w_bot[0].len()];
        self.d_top.iter_mut().for_each(|d| *d = 0.0);
        self.d_bot.iter_mut().for_each(|d| *d = 0.0);
        for k in 0..CONTACTS {
            let sep = offset + self.top.displacement(&self.w_top[k])
                - self.bottom.displacement(&self.w_bot[k]);
            if sep < 0.0 {
                let (vt, vb) = (
                    self.top.velocity(&self.w_top[k]) + v_rigid,
                    self.bottom.velocity(&self.w_bot[k]),
                );
                let f = (CONTACT_K * (-sep) * (-sep).sqrt() - CONTACT_DAMP * (vt - vb)).max(0.0);
                for (i, (a, b)) in self.w_top[k].iter().zip(&self.w_bot[k]).enumerate() {
                    f_top[i] += f * a;
                    f_bot[i] -= f * b;
                    if self.latched {
                        self.d_top[i] += PRESS_DAMP * a * a;
                        self.d_bot[i] += PRESS_DAMP * b * b;
                    }
                }
                f_rigid += f;
            }
        }
        if !self.latched {
            self.vz += f_rigid / TOP_MASS;
            self.z += self.vz;
        }
        self.top.damp_raw(&self.d_top);
        self.bottom.damp_raw(&self.d_bot);
        self.top.push_raw(&f_top);
        self.bottom.push_raw(&f_bot);
        // The bottom plate is shielded by the top one: heard a little
        // less.
        (self.top.tick(rng) + 0.7 * self.bottom.tick(rng)) * self.p.level
    }
}

/// A 14" pair: stiff, high, quick.
pub fn default_params() -> HiHatParams {
    HiHatParams {
        plate: CymbalParams {
            stiffness: 1.0,
            dome: 260.0,
            decay: 1.5,
            // Barely any: a hat's high partials are the ones that ring.
            hf_damp: 0.03,
            strike_pos: 0.6,
            hardness: 0.95,
            mallet: 1.0,
            nonlin: 0.0,
            level: 1.0,
        },
        gap: 0.4,
        press: 1.0,
        level: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze;
    use crate::util::SR;

    fn render(h: &mut HiHat, secs: f32) -> Vec<f32> {
        let mut rng = Rng(3);
        (0..(secs * SR) as usize)
            .map(|_| h.tick(&mut rng))
            .collect()
    }

    #[test]
    fn pedal_chokes_and_chicks() {
        let modes = Arc::new(Modes::compute());
        // Open: rings. Closed: the same hit dies fast.
        let mut open = HiHat::new(modes.clone(), default_params());
        open.strike(1.0);
        let o = render(&mut open, 1.0);
        let mut closed = HiHat::new(modes.clone(), default_params());
        closed.pedal(true);
        let _ = render(&mut closed, 0.1);
        closed.strike(1.0);
        let c = render(&mut closed, 1.0);
        let tail = |b: &[f32]| {
            analyze::rms(analyze::win(b, 0.3, 0.6)) / analyze::rms(analyze::win(b, 0.0, 0.05))
        };
        assert!(o.iter().all(|s| s.is_finite()) && c.iter().all(|s| s.is_finite()));
        assert!(
            tail(&c) < 0.3 * tail(&o),
            "closed hat rings: open tail {:.3}, closed tail {:.3}",
            tail(&o),
            tail(&c)
        );
        // The foot alone makes a sound.
        let mut chick = HiHat::new(modes, default_params());
        let silent = analyze::rms(&render(&mut chick, 0.1));
        chick.pedal(true);
        let heard = analyze::rms(&render(&mut chick, 0.1));
        assert!(heard > 20.0 * silent.max(1e-6), "no chick: {heard:.4}");
    }
}
