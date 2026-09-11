//! timber — physical modeling explorations.
//!
//! Three models, all the same idea — an exciter driving a resonator:
//!   string: noise burst  → delay-line loop       (string.rs)
//!   modal:  strike       → bank of ringing modes (modal.rs)
//!   voice:  glottal buzz → moving formant modes  (voice.rs)

pub mod drums;
pub mod modal;
pub mod mouth;
pub mod stream;
pub mod string;
pub mod util;
pub mod voice;
