//! Speech, stage 3: singing.
//!
//! A song is speech with two extras: each syllable carries a pitch and
//! a duration. The vowel nucleus stretches to fill the note, pitch
//! glides across the consonant transitions (free portamento — which is
//! what legato singing is), and everything else — coarticulation,
//! allophony, the velum — is the same machinery speech uses.
//!
//! The reference pitch is 120 Hz: the desk's voice pitch control acts
//! as a transpose.
//!
//! And, non-negotiably, the flagship score is "Daisy Bell": Kelly,
//! Lochbaum and Max Mathews made an IBM 704 sing it in 1961 with
//! exactly this architecture — the performance HAL quotes as he dies.
//! Sixty-five years later, a hobby tract returns the favor.

use crate::speak::{Seg, phrase};

/// One sung syllable: (phonemes, pitch Hz at reference, beats).
pub type Syllable<'a> = (&'a str, f32, f32);

/// Compile a score into a Seg stream for speak::Utterance.
pub fn song(score: &[Syllable], bpm: f32) -> Vec<Seg> {
    let spb = 60.0 / bpm;
    let mut out = Vec::new();
    for &(phon, freq, beats) in score {
        let mut segs = phrase(phon);
        if segs.is_empty() {
            continue;
        }
        let target = beats * spb;
        let total: f32 = segs.iter().map(|s| s.trans + s.hold).sum();
        if target > total {
            // Stretch the nucleus: the longest voiced, open segment is
            // the vowel that carries the note.
            let idx = segs
                .iter()
                .enumerate()
                .filter(|(_, s)| s.voiced > 0.5 && s.flow > 0.0)
                .max_by(|a, b| a.1.hold.total_cmp(&b.1.hold))
                .map(|(i, _)| i)
                .unwrap_or(0);
            segs[idx].hold += target - total;
        } else {
            // Note shorter than the syllable speaks: compress evenly.
            let k = target / total.max(1e-6);
            for s in &mut segs {
                s.trans *= k;
                s.hold *= k;
            }
        }
        for s in &mut segs {
            s.freq = freq;
        }
        out.extend(segs);
    }
    out
}

// A2..G3 in Hz.
const A2: f32 = 110.0;
const B2: f32 = 123.47;
const C3: f32 = 130.81;
const D3: f32 = 146.83;
const E3: f32 = 164.81;
const F3: f32 = 174.61;
const G3: f32 = 196.0;
const G2: f32 = 98.0;

/// The chorus of "Daisy Bell" (Harry Dacre, 1892), hand-phonemized.
/// 3/4 time, brisk waltz; the transcription is traditional, lightly
/// simplified.
pub const DAISY: &[Syllable] = &[
    ("d e i", G3, 3.0),
    ("z i", E3, 3.0),
    ("d e i", C3, 3.0),
    ("z i", G2, 3.0),
    ("g i v", A2, 1.0),
    ("m i", B2, 1.0),
    ("y o r", C3, 1.0),
    ("a n", A2, 2.0),
    ("s e r", G2, 1.0),
    ("d u", D3, 3.0),
    (".", G2, 1.0),
    ("a i m", E3, 1.5),
    ("h a f", G3, 1.5),
    ("k r e i", E3, 1.5),
    ("z i", C3, 1.5),
    ("o l", D3, 1.0),
    ("f o r", G3, 1.0),
    ("d e", F3, 1.0),
    ("l a v", E3, 1.0),
    ("o v", D3, 1.0),
    ("y u", C3, 3.0),
];

pub fn daisy() -> Vec<Seg> {
    song(DAISY, 138.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speak::Utterance;
    use crate::tract::{Params, Tract};
    use crate::util::{Rng, SR, measured_freq};

    fn base() -> Params {
        Params {
            freq: 120.0, // transpose 1.0
            tongue_pos: 0.5,
            constrict: 0.3,
            lips: 0.8,
            velum: 0.0,
            tip: 0.0,
            voiced: 1.0,
            breath: 0.05,
            flow: 1.0,
            vibrato: 0.0,
            level: 1.0,
            gate: false,
        }
    }

    fn render(segs: Vec<Seg>) -> Vec<f32> {
        let mut u = Utterance::new(segs).unwrap();
        let mut tract = Tract::new();
        let mut rng = Rng(9);
        let b = base();
        let mut out = Vec::new();
        while let Some(p) = u.step(&b) {
            out.push(tract.tick(&p, &mut rng));
        }
        out
    }

    #[test]
    fn song_runs_to_the_score() {
        let expect: f32 = DAISY.iter().map(|(_, _, b)| b).sum::<f32>() * 60.0 / 138.0;
        let out = render(daisy());
        assert!(out.iter().all(|s| s.is_finite()));
        let got = out.len() as f32 / SR;
        assert!(
            (got - expect).abs() < 0.05,
            "song {got:.2}s vs score {expect:.2}s"
        );
    }

    #[test]
    fn sung_note_holds_its_pitch() {
        // One syllable on G3 for three beats: the held vowel must sit
        // at the note, not at the speaking pitch.
        let out = render(song(&[("d e i", G3, 3.0)], 84.0));
        let n = out.len();
        let held = &out[n / 2..n / 2 + 44100.min(n / 2)];
        let f = measured_freq(held, G3);
        let cents = 1200.0 * (f / G3).log2();
        assert!(cents.abs() < 35.0, "sung pitch {f:.1} Hz ({cents:+.1}c)");
    }
}
