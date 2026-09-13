//! The disc every struck surface lives on: a circular mask of cells
//! on a square grid with a border for the stencils, optionally with a
//! clamped center (a cymbal's stand). One definition, so the membrane
//! (mesh.rs), the finite-difference plate (plate.rs) and the modal
//! cymbal (cymbal.rs, whose mode shapes come from the same disc) can't
//! drift apart on radius, rim or clamp.

pub struct Disc {
    /// Grid width including the border.
    pub w: usize,
    /// In the domain: inside the disc (the clamp included — its cells
    /// are held at zero but count as neighbours).
    pub mask: Vec<bool>,
    /// Cells that move: in the disc and outside the clamp.
    pub cells: Vec<usize>,
    /// Center coordinate (the same on both axes).
    pub center: f32,
    pub r: usize,
    pub clamp: f32,
}

impl Disc {
    /// Radius `r` in cells, `border` cells around it (1 for a 5-point
    /// stencil, 2 for the 13-point biharmonic), and a clamp radius
    /// (0 for none). The rim is the staircase of cells within r + ½.
    pub fn new(r: usize, border: usize, clamp: f32) -> Self {
        let w = 2 * r + 1 + 2 * border;
        let center = (r + border) as f32;
        let mut mask = vec![false; w * w];
        let mut cells = Vec::new();
        for j in border..w - border {
            for i in border..w - border {
                let (dx, dy) = (i as f32 - center, j as f32 - center);
                let r2 = dx * dx + dy * dy;
                if r2 <= (r as f32 + 0.5) * (r as f32 + 0.5) {
                    mask[j * w + i] = true;
                    // A zero clamp clamps nothing — not even the
                    // center cell, whose r² is exactly 0.
                    if clamp <= 0.0 || r2 > clamp * clamp {
                        cells.push(j * w + i);
                    }
                }
            }
        }
        Disc {
            w,
            mask,
            cells,
            center,
            r,
            clamp,
        }
    }

    /// Cell index at integer offsets from the center.
    pub fn at(&self, dx: isize, dy: isize) -> usize {
        let c = self.center as isize;
        ((c + dy) as usize) * self.w + (c + dx) as usize
    }

    /// Where a stick lands for a strike position 0 (just outside the
    /// clamp) .. 1 (a cell in from the rim), along +x from the center.
    pub fn strike_center(&self, pos: f32) -> (f32, f32) {
        let inner = self.clamp;
        let span = (self.r as f32 - inner - 1.0).max(0.0);
        (
            self.center + inner + pos.clamp(0.0, 1.0) * span,
            self.center,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_is_round_bordered_and_clamped() {
        let d = Disc::new(10, 2, 1.5);
        assert_eq!(d.w, 25);
        // Every masked cell keeps a full border of unmasked cells.
        for j in 0..d.w {
            for i in 0..d.w {
                if d.mask[j * d.w + i] {
                    assert!(i >= 2 && i < d.w - 2 && j >= 2 && j < d.w - 2);
                }
            }
        }
        // The clamp is in the mask but not among the moving cells.
        let c = d.at(0, 0);
        assert!(d.mask[c] && !d.cells.contains(&c));
        assert!(d.cells.contains(&d.at(5, 0)));
        // Area ≈ π(r + ½)².
        let area = d.mask.iter().filter(|&&m| m).count() as f32;
        assert!((area / (std::f32::consts::PI * 10.5 * 10.5) - 1.0).abs() < 0.05);
        let no_clamp = Disc::new(12, 1, 0.0);
        assert_eq!(
            no_clamp.cells.len(),
            no_clamp.mask.iter().filter(|&&m| m).count()
        );
    }
}
