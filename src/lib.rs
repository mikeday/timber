//! timber — physical modeling explorations.
//!
//! Three models, all the same idea — an exciter driving a resonator:
//!   string: noise burst  → delay-line loop       (string.rs)
//!   modal:  strike       → bank of ringing modes (modal.rs)
//!   voice:  glottal buzz → moving formant modes  (voice.rs)

pub mod analyze;
pub mod body;
pub mod cymbal;
pub mod drums;
pub mod grid;
pub mod looper;
pub mod mesh;
pub mod modal;
pub mod mouth;
pub mod plate;
pub mod sing;
pub mod speak;
pub mod stick;
pub mod stream;
pub mod string;
pub mod tract;
pub mod util;
pub mod voice;
