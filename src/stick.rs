//! The stick: a mass thrown at a surface through a Hertz contact
//! spring. Contact duration emerges — and shrinks as the hit gets
//! harder, so hard hits crack and soft hits thud, the way a real
//! instrument's brightness follows the player's arm. Shared by the
//! membrane (mesh.rs) and the plate (plate.rs).

/// Stick mass in units of the contact patch's mass — a good fraction
/// of the whole head, as a real stick is. It is the surface's own
/// stiffness, not the patch, that throws the stick back, so contact
/// time is set by this mass against that stiffness once the tip is
/// hard.
const STICK_MASS: f32 = 25.0;
/// Contact stiffness across the hardness range (felt beater .. wood
/// tip), in per-step² units. Measured contact: felt beater on a 55 Hz
/// kick ~15 ms, wood tip on a 165 Hz tom ~2 ms, shortening with
/// strength (examples/stickprobe.rs).
const K_SOFT: f32 = 0.05;
const K_HARD: f32 = 6.0;

/// Hertz contact stiffness for a hardness in 0 (felt) .. 1 (wood).
pub fn contact_k(hardness: f32) -> f32 {
    K_SOFT * (K_HARD / K_SOFT).powf(hardness.clamp(0.0, 1.0))
}

/// The contact footprint on a masked grid: a raised-cosine patch,
/// `(cell, weight)` with weights summing to 1, and the patch's mass
/// (the raw bump sum, in cell masses).
pub fn patch(
    mask: &[bool],
    width: usize,
    center: (f32, f32),
    rad: f32,
) -> (Vec<(usize, f32)>, f32) {
    let mut cells = Vec::new();
    let mut mass = 0.0f32;
    for j in 1..width - 1 {
        for i in 1..width - 1 {
            let idx = j * width + i;
            let (dx, dy) = (i as f32 - center.0, j as f32 - center.1);
            let d = (dx * dx + dy * dy).sqrt();
            if mask[idx] && d < rad {
                let bump = 0.5 * (1.0 + (std::f32::consts::PI * d / rad).cos());
                cells.push((idx, bump));
                mass += bump;
            }
        }
    }
    for (_, w) in &mut cells {
        *w /= mass.max(1e-9);
    }
    (cells, mass.max(1e-9))
}

#[derive(Clone, Copy, Default)]
pub struct Stick {
    /// Position and velocity in surface-displacement units.
    z: f32,
    v: f32,
    on: bool,
    /// Internal steps the last throw spent in contact (diagnostic).
    contact_steps: u32,
}

impl Stick {
    /// Throw the stick at the surface: it starts touching (`under` is
    /// the surface displacement beneath it) with this velocity.
    pub fn throw(&mut self, under: f32, velocity: f32) {
        self.z = under;
        self.v = velocity;
        self.on = true;
        self.contact_steps = 0;
    }

    /// One step. Hertz law, force ∝ compression^1.5: the surface takes
    /// the returned force over its patch, the stick takes the
    /// reaction; when the surface has pushed it back off, it is gone.
    /// `patch_mass` is the patch's mass in cell masses.
    pub fn step(&mut self, under: f32, hardness: f32, patch_mass: f32) -> f32 {
        if !self.on {
            return 0.0;
        }
        let c = self.z - under;
        let mut force = 0.0;
        if c > 0.0 {
            force = contact_k(hardness) * c * c.sqrt();
            self.contact_steps += 1;
        } else if self.v < 0.0 {
            self.on = false;
        }
        self.v -= force / (STICK_MASS * patch_mass);
        self.z += self.v;
        force
    }

    pub fn on(&self) -> bool {
        self.on
    }

    pub fn contact_steps(&self) -> u32 {
        self.contact_steps
    }
}
