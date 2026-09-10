//! timber — physical modeling explorations.
//!
//! Three models, all the same idea — an exciter driving a resonator:
//!   string: noise burst  → delay-line loop       (string.rs)
//!   modal:  strike       → bank of ringing modes (modal.rs)
//!   voice:  glottal buzz → moving formant modes  (voice.rs)

pub mod modal;
pub mod string;
pub mod util;
pub mod voice;
