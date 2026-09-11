//! Speech, stage 1: the gesture sequencer.
//!
//! Speech is timed articulation: each phoneme is a tract target (tongue,
//! constriction, lips) plus voicing, aspiration, and timing, and a word
//! is a list of them glided through in order. Nothing new is synthesized
//! here — the sequencer just drives tract::Tract the way the pad does,
//! but with a schedule instead of a finger.
//!
//! Plosives are two gestures: a closure (silent for /p t k/, murmured
//! for /b d g/) and a release — a burst of aspiration before voicing
//! resumes. That unvoiced gap is the voice-onset time, and it is most of
//! what separates "pa" from "ba".
//!
//! The phoneme strings are whitespace-separated tokens, e.g.
//! "h e l o . w a w". Unknown tokens are skipped. "." is a pause.

use crate::tract::{Params, VOWELS};
use crate::util::SR;

#[derive(Clone, Copy)]
pub struct Seg {
    pub tongue: f32,
    pub constrict: f32,
    pub lips: f32,
    pub voiced: f32,
    pub breath: f32,
    pub flow: f32,
    /// Velum opening: 1.0 reroutes sound through the nose (m, n, ng).
    pub velum: f32,
    /// Tongue-tip raising: /n t d l/ close with the tip, body free.
    pub tip: f32,
    /// Seconds gliding from the previous target...
    pub trans: f32,
    /// ...then seconds holding this one.
    pub hold: f32,
}

fn seg(
    (tongue, constrict, lips): (f32, f32, f32),
    voiced: f32,
    breath: f32,
    trans_ms: f32,
    hold_ms: f32,
) -> Seg {
    Seg {
        tongue,
        constrict,
        lips,
        voiced,
        breath,
        flow: 1.0,
        velum: 0.0,
        tip: 0.0,
        trans: trans_ms / 1000.0,
        hold: hold_ms / 1000.0,
    }
}

/// A stop closure: air pressure with no flow.
fn closed(s: Seg) -> Seg {
    Seg { flow: 0.0, ..s }
}

fn vowel(name: &str) -> (f32, f32, f32) {
    let v = VOWELS.iter().find(|v| v.0 == name).unwrap();
    (v.1, v.2, v.3)
}

/// A neutral, open, schwa-ish tract.
const NEUTRAL: (f32, f32, f32) = (0.5, 0.35, 0.8);

/// `asp` is the voiceless release's aspiration level — per place,
/// because geometry amplifies differently: a /p/ puff exits at the
/// barely-open lips with no tract in front to filter it, and a /k/
/// puff excites the whole tube; /t/ releases through the front
/// constriction and can afford more.
fn plosive(place: (f32, f32, f32), voiced: bool, asp: f32) -> Vec<Seg> {
    let murmur = if voiced { 0.45 } else { 0.0 };
    vec![
        // Closure: the tube shuts at the place of articulation.
        closed(seg(
            place,
            murmur,
            0.0,
            35.0,
            if voiced { 48.0 } else { 75.0 },
        )),
        // Release: the burst itself is the tube's own frication
        // transient as the closure sweeps open — the added glottal
        // aspiration is only a whisper on top (medial stops are barely
        // aspirated; doubling the burst with breath noise reads as
        // hiss). Voiceless stops still keep the glottis off through
        // the gap: the VOT.
        if voiced {
            seg(NEUTRAL, 1.0, 0.10, 12.0, 15.0)
        } else {
            seg(NEUTRAL, 0.0, asp, 12.0, 24.0)
        },
    ]
}

/// A nasal: the mouth closes at the place of articulation while the
/// velum opens — the murmur radiates from the nostrils, with the sealed
/// oral cavity as the anti-resonant side branch.
///
/// Three gestures, because a tongue does not SLIDE a closure along the
/// palate (our single-hump tongue would: a deep seal sweeping through
/// an energized tube thumps once forming and once releasing — the
/// nasal double-click). Instead: approach the place at vowel depth,
/// press to seal IN PLACE, lift IN PLACE — the following vowel then
/// carries the tongue home at gentle vowel pace.
fn nasal(place: (f32, f32, f32), approach_velum: f32) -> Vec<Seg> {
    let (tp, con, lips) = place;
    vec![
        // Approach: travel there SHALLOW — below vowel depth, so the
        // transit reads as a glide, not a vowel (the i→velar route
        // passes straight through /u/'s position; at /u/'s depth it
        // says "u") — and briskly, with the velum barely open (a
        // nasalized transit is a stray /n/: "si-nung").
        Seg {
            velum: approach_velum,
            ..seg((tp, con.min(0.55), lips), 1.0, 0.02, 22.0, 4.0)
        },
        // Seal: press down where you stand. (Crisp is fine: the press
        // crosses fricative geometry, but the open velum has diverted
        // the airflow, so nothing fricates — see tract.rs oral_flow.)
        Seg {
            velum: 1.0,
            ..seg((tp, con, lips), 1.0, 0.02, 24.0, 84.0)
        },
        // Lift: release where you stand; the velum shuts just after,
        // quickly — lingering nasality glides the vowel onset ("nya").
        Seg {
            velum: 0.02,
            ..seg((tp, con.min(0.5), lips), 1.0, 0.04, 20.0, 8.0)
        },
    ]
}

/// A tip nasal (/n/): the tip flicks up in place while the body —
/// coarticulated by phrase() — holds the neighboring vowel throughout.
/// No body travel, no /j/-transit.
fn nasal_tip() -> Vec<Seg> {
    vec![
        Seg {
            velum: 1.0,
            tip: 1.0,
            ..seg(NEUTRAL, 1.0, 0.02, 30.0, 95.0)
        },
        Seg {
            velum: 0.02,
            tip: 0.0,
            ..seg(NEUTRAL, 1.0, 0.04, 22.0, 8.0)
        },
    ]
}

/// A tip plosive (/t d/): closure by tip flick, body free.
fn plosive_tip(voiced: bool, asp: f32) -> Vec<Seg> {
    let murmur = if voiced { 0.45 } else { 0.0 };
    vec![
        closed(Seg {
            tip: 1.0,
            ..seg(NEUTRAL, murmur, 0.0, 30.0, if voiced { 48.0 } else { 70.0 })
        }),
        if voiced {
            Seg {
                tip: 0.0,
                ..seg(NEUTRAL, 1.0, 0.10, 12.0, 15.0)
            }
        } else {
            Seg {
                tip: 0.0,
                ..seg(NEUTRAL, 0.0, asp, 12.0, 24.0)
            }
        },
    ]
}

/// A velar plosive (/k g/): the body IS the closer, so it approaches
/// shallow and presses in place — a deep closure formed mid-travel
/// thumps (the same sweep-click the nasals once had).
fn plosive_velar(voiced: bool, asp: f32) -> Vec<Seg> {
    let murmur = if voiced { 0.45 } else { 0.0 };
    vec![
        // The approach is silent — a voiced, half-open tube for 30 ms
        // is a schwa prepended to the stop ("go" becomes "ago").
        closed(seg((0.3, 0.7, 0.9), 0.0, 0.0, 28.0, 4.0)),
        closed(seg(
            (0.3, 1.15, 0.9),
            murmur,
            0.0,
            20.0,
            if voiced { 38.0 } else { 46.0 },
        )),
        if voiced {
            Seg {
                tip: 0.0,
                ..seg(NEUTRAL, 1.0, 0.10, 16.0, 15.0)
            }
        } else {
            Seg {
                tip: 0.0,
                ..seg(NEUTRAL, 0.0, asp, 16.0, 24.0)
            }
        },
    ]
}

fn phoneme(tok: &str) -> Option<Vec<Seg>> {
    Some(match tok {
        "a" => vec![seg(vowel("ah"), 1.0, 0.04, 70.0, 160.0)],
        "e" => vec![seg(vowel("eh"), 1.0, 0.04, 70.0, 150.0)],
        "i" => vec![seg(vowel("ee"), 1.0, 0.04, 70.0, 150.0)],
        "o" => vec![seg(vowel("oh"), 1.0, 0.04, 70.0, 150.0)],
        "u" => vec![seg(vowel("oo"), 1.0, 0.04, 70.0, 150.0)],
        // Semivowels: vowels visited briefly.
        "w" => vec![seg(vowel("oo"), 1.0, 0.04, 50.0, 35.0)],
        "y" => vec![seg(vowel("ee"), 1.0, 0.04, 50.0, 35.0)],
        "h" => vec![seg(NEUTRAL, 0.0, 0.20, 40.0, 70.0)],
        // Crude approximants — no lateral or true rhotic in a plain
        // tube, but these gestures read as l/r in context.
        // A lateral needs edges a plain tube lacks: approximate with a
        // quick front contact and an abrupt release, lips spread the
        // whole time (a slow rounded glide out of /l/ IS a /w/).
        // /l/: near-contact of the tip (a true lateral needs side
        // passages a tube lacks, but the in-place tip flick carries
        // most of the percept), body free.
        "l" => vec![
            Seg {
                tip: 0.85,
                ..seg(NEUTRAL, 1.0, 0.03, 22.0, 55.0)
            },
            Seg {
                tip: 0.0,
                ..seg(NEUTRAL, 1.0, 0.03, 12.0, 8.0)
            },
        ],
        "r" => vec![seg((0.45, 0.62, 0.45), 1.0, 0.03, 60.0, 70.0)],
        // Fricatives: tight constrictions past the pad's clamp; the
        // turbulence and its coloring come from the tube itself.
        // Nearly zero glottal breath: a fricative's pressure drops at
        // the constriction — glottal noise underneath smears an /h/
        // into the hiss.
        "s" => vec![seg((0.85, 0.96, 0.9), 0.0, 0.03, 60.0, 130.0)],
        "sh" => vec![seg((0.55, 0.94, 0.55), 0.0, 0.03, 60.0, 130.0)],
        "f" => vec![seg((0.5, 0.4, 0.02), 0.0, 0.05, 50.0, 110.0)],
        "z" => vec![Seg {
            flow: 0.7, // voicing throttles the air supply
            ..seg((0.85, 0.95, 0.9), 0.6, 0.04, 60.0, 120.0)
        }],
        "m" => nasal((0.5, 0.35, 0.0), 1.0),
        "n" => nasal_tip(),
        "ng" => nasal((0.3, 1.15, 0.9), 0.1),
        "." => vec![closed(seg(NEUTRAL, 0.0, 0.0, 60.0, 160.0))],
        "p" => plosive((0.5, 0.35, 0.0), false, 0.08),
        "b" => plosive((0.5, 0.35, 0.0), true, 0.0),
        "t" => plosive_tip(false, 0.12),
        "d" => plosive_tip(true, 0.0),
        "k" => plosive_velar(false, 0.08),
        "g" => plosive_velar(true, 0.0),
        _ => return None,
    })
}

/// Parse a phoneme string; unknown tokens are skipped.
pub fn phrase(text: &str) -> Vec<Seg> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    let mut segs: Vec<Seg> = Vec::new();
    let mut after_nasal = false;
    for (i, tok) in toks.iter().enumerate() {
        let Some(mut ph) = phoneme(tok) else { continue };
        // Coming off a nasal, the tongue must travel home FAST — after
        // /n/ a leisurely 70 ms descent from the palatal position is a
        // guided tour through /j/-space ("na" becomes "nya").
        if after_nasal
            && let Some(first) = ph.first_mut()
            && first.flow > 0.0
        {
            first.trans = first.trans.min(0.03);
        }
        after_nasal = matches!(*tok, "m" | "n" | "ng");
        // A word-final nasal ends IN the murmur: releasing into
        // silence is a voiced schwa ("sing" grows an "-er"). Drop the
        // lift and let the phrase-end fade finish it.
        if after_nasal && i + 1 == toks.len() && ph.len() > 1 {
            ph.truncate(ph.len() - 1);
        }
        // Word-final voiced stops are unreleased too: a voiced release
        // into silence is a schwa ("kab" grows a "-beh"). Voiceless
        // finals keep their small burst — that reads correctly. The
        // unreleased closure must SNAP shut: with no burst, the abrupt
        // cut of the vowel is the stop's only remaining signature.
        if i + 1 == toks.len() && matches!(*tok, "b" | "d" | "g") && ph.len() > 1 {
            ph.truncate(ph.len() - 1);
            if let Some(clo) = ph.last_mut() {
                clo.trans = clo.trans.min(0.022);
            }
        }
        // /h/ is a voiceless onset of whatever follows — "ha" is
        // breathed through the /a/ shape, "hee" through /i/ — not a
        // hiss at some neutral posture.
        if *tok == "h"
            && let Some(next) = toks.get(i + 1).and_then(|t| phoneme(t))
            && let (Some(h), Some(n)) = (ph.first_mut(), next.first())
            // ...but not a stop closure: breathing into a sealed tube
            // is silence, not an onset.
            && n.flow > 0.0
        {
            h.tongue = n.tongue;
            // Laxer than the vowel proper: /a/'s full pharyngeal pinch
            // sits inside the turbulence zone, masked when voiced but a
            // jet when whispered.
            h.constrict = n.constrict.min(0.72);
            h.lips = n.lips;
        }
        // Consonants coarticulate: whatever articulator is NOT doing
        // the closing holds the next phoneme's full posture throughout
        // (that anticipation is the place cue — without it every nasal
        // reads alveolar and vowels form from a dark neutral tube).
        // Tip consonants free the entire tongue body and lips; /m/
        // frees the tongue; /ng/ frees the lips.
        let tip_based = matches!(*tok, "n" | "t" | "d" | "l");
        let velar = matches!(*tok, "ng" | "k" | "g");
        // Coarticulation anchor: the next phoneme if there is one, else
        // CARRYOVER from the previous — a final consonant holds the
        // outgoing vowel's posture (word-final ng after /o/ must stay
        // rounded, not snap to a default).
        if (tip_based || velar || matches!(*tok, "m" | "p" | "b"))
            && let Some(nx) = toks
                .get(i + 1)
                .and_then(|t| phoneme(t))
                .and_then(|n| n.first().copied())
                .filter(|nx| nx.flow > 0.0)
                .or_else(|| segs.last().copied())
        {
            let alveolar_plosive = matches!(*tok, "t" | "d");
            // Allophonic velar fronting: the velar closure meets the
            // neighboring vowel partway — "king"'s velar is nearly
            // palatal, "kong"'s is far back. Real English does this,
            // and it is what keeps the body transit short enough not
            // to sound out a stray vowel en route ("si-nung").
            // (Weight 0.35: at 0.5 the fronted allophone reaches
            // palatal territory and "sing" grows an Italian gli.)
            let velar_pos = 0.3 + 0.35 * (nx.tongue - 0.3);
            for sgm in ph.iter_mut() {
                if velar {
                    sgm.tongue = velar_pos;
                } else if alveolar_plosive {
                    // The tip drags the connected body halfway forward:
                    // that partial fronting IS the alveolar F2 locus. A
                    // fully vowel-parked body releases with a low F2 —
                    // the labial signature — and /d/ reads as /b/.
                    sgm.tongue = 0.5 * (nx.tongue + 0.75);
                    sgm.constrict = nx.constrict.min(0.6);
                } else if tip_based || matches!(*tok, "m" | "p" | "b") {
                    // Labials free the whole tongue (m is the model
                    // citizen: without this the neutral mid body gives
                    // an ambiguous, d-ish F2 locus).
                    sgm.tongue = nx.tongue;
                    sgm.constrict = nx.constrict.min(0.85);
                }
                if tip_based || velar {
                    sgm.lips = nx.lips;
                }
            }
            if let Some(rel) = ph.last_mut() {
                if matches!(*tok, "m" | "p" | "b") {
                    rel.lips = nx.lips;
                } else if *tok == "ng" {
                    // The velar body departs as it lifts.
                    rel.tongue = 0.5 * (rel.tongue + nx.tongue);
                }
            }
        }
        segs.extend(ph);
    }
    // No explicit phrase-final devoicing needed: the tract holds its
    // articulation whenever the gate is off, so the release after the
    // last segment fades the source in place (the engine-level fix for
    // the phantom-syllable family).
    segs
}

/// A scheduled utterance, stepped once per output sample.
pub struct Utterance {
    segs: Vec<Seg>,
    i: usize,
    t: f32,
    /// The segment being glided FROM — simply the previous Seg.
    from: Seg,
}

impl Utterance {
    pub fn new(segs: Vec<Seg>) -> Option<Self> {
        let first = *segs.first()?;
        Some(Utterance {
            from: first,
            segs,
            i: 0,
            t: 0.0,
        })
    }

    /// The tract parameters for this sample, or None when the utterance
    /// has finished. `base` supplies pitch, level and vibrato — the
    /// performance; the schedule supplies the articulation.
    pub fn step(&mut self, base: &Params) -> Option<Params> {
        loop {
            let s = *self.segs.get(self.i)?;
            let total = s.trans + s.hold;
            if self.t >= total {
                self.t -= total;
                self.from = s;
                self.i += 1;
                continue;
            }
            let u = if s.trans <= 0.0 {
                1.0
            } else {
                (self.t / s.trans).min(1.0)
            };
            let lerp = |a: f32, b: f32| a + (b - a) * u;
            self.t += 1.0 / SR;
            return Some(Params {
                freq: base.freq,
                tongue_pos: lerp(self.from.tongue, s.tongue),
                constrict: lerp(self.from.constrict, s.constrict),
                lips: lerp(self.from.lips, s.lips),
                velum: lerp(self.from.velum, s.velum),
                tip: lerp(self.from.tip, s.tip),
                voiced: lerp(self.from.voiced, s.voiced),
                breath: lerp(self.from.breath, s.breath),
                flow: lerp(self.from.flow, s.flow),
                vibrato: base.vibrato * 0.5,
                level: base.level,
                gate: true,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tract::Tract;
    use crate::util::Rng;

    fn render(text: &str) -> Vec<f32> {
        let mut u = Utterance::new(phrase(text)).unwrap();
        let mut tract = Tract::new();
        let mut rng = Rng(9);
        let base = Params {
            freq: 120.0,
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
        };
        let mut out = Vec::new();
        while let Some(p) = u.step(&base) {
            out.push(tract.tick(&p, &mut rng));
        }
        out
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|s| s * s).sum::<f32>() / buf.len().max(1) as f32).sqrt()
    }

    #[test]
    fn utterance_runs_to_schedule_and_ends() {
        let segs = phrase("a b a");
        let expect: f32 = segs.iter().map(|s| s.trans + s.hold).sum();
        let out = render("a b a");
        assert!(out.iter().all(|s| s.is_finite()));
        let got = out.len() as f32 / SR;
        assert!(
            (got - expect).abs() < 0.01,
            "duration {got:.3}s vs schedule {expect:.3}s"
        );
    }

    #[test]
    fn plosive_closure_dips_the_amplitude() {
        // "a b a": the lips shut mid-word — a real amplitude dip must
        // separate the vowels.
        let out = render("a b a");
        let n = out.len();
        let vowel1 = rms(&out[(0.10 * SR) as usize..(0.20 * SR) as usize]);
        // Quietest 25 ms window in the middle half.
        let mut dip = f32::MAX;
        let (lo, hi) = (n / 4, 3 * n / 4);
        for w in out[lo..hi].chunks((0.025 * SR) as usize) {
            dip = dip.min(rms(w));
        }
        let vowel2 = rms(&out[n - (0.15 * SR) as usize..]);
        assert!(vowel1 > 0.02, "first vowel too quiet: {vowel1}");
        assert!(vowel2 > 0.02, "second vowel too quiet: {vowel2}");
        assert!(
            dip < vowel1 * 0.25,
            "no closure dip: {dip} vs vowel {vowel1}"
        );
    }

    #[test]
    fn nasal_murmurs_where_a_stop_is_silent() {
        // "a m a" vs "a b a": both close the lips mid-word, but the
        // nasal's open velum keeps the hum radiating from the nose —
        // its middle must carry real energy where the stop's is a dip.
        // Windows anchored inside the closures via the schedule itself
        // (a fixed-fraction window once counted /b/'s voiced release —
        // and, embarrassingly, the old click artifacts — as murmur).
        let mid_rms = |text: &str| {
            let segs = phrase(text);
            let vowel = segs[0].trans + segs[0].hold;
            let out = render(text);
            let (lo, hi) = (
                ((vowel + 0.035) * SR) as usize,
                ((vowel + 0.080) * SR) as usize,
            );
            rms(&out[lo..hi])
        };
        let m = mid_rms("a m a");
        let b = mid_rms("a b a");
        assert!(
            m > 0.02 && m > b * 2.0,
            "no murmur: m-mid {m:.4} vs b-mid {b:.4}"
        );
    }

    #[test]
    fn fricative_is_noisier_than_vowels() {
        // "a s a": the middle must be dominated by high-frequency noise.
        // Zero-crossing rate, not spectral slope — the lip-radiation
        // pre-emphasis tilts every segment bright, but an /s/ still
        // oscillates several times faster than an /a/.
        let zcr = |buf: &[f32]| {
            buf.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f32
                / buf.len() as f32
        };
        let out = render("a s a");
        let n = out.len();
        let vowel = zcr(&out[(0.10 * SR) as usize..(0.20 * SR) as usize]);
        let mid = zcr(&out[n / 2 - 2205..n / 2 + 2205]);
        assert!(
            mid > vowel * 2.0,
            "s not noisy: mid zcr {mid:.4} vs vowel {vowel:.4}"
        );
    }
}
