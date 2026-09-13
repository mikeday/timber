//! The timber desk: a little realtime mixing desk over the models.
//!
//! Architecture: every instrument now streams — the audio thread owns
//! the string bank (timber::stream), the drum kit (timber::drums) and
//! the mouth (timber::mouth), and the UI steers them live through
//! atomics and messages: pluck and bow strings, strike pads whose knobs
//! act on sounds mid-ring, hold the mouth and sweep its vowel plane.
//! Only one-shot sung syllables are still rendered as buffers with the
//! offline voice code. The mixer (gain, pan, mute, master with a tanh
//! limiter) is always live.
//!
//! The voice column runs one of two engines (checkbox): the formant
//! mouth with its F1×F2 vowel plane, or the Kelly-Lochbaum tract with a
//! tongue surface, a live tube-profile drawing, and a phoneme box that
//! speaks through timber::speak.
//!
//! Keys: Z X C V B N M , . 8 9 = drums (shift = roll) · 1 2 3 4 5 = mesh drums
//! · 6 7 = cymbals (shift = hard hit) · A S D F G H J K =
//! pluck strings (finger while bowing; melody while a voice engine is
//! held or speaking) · Q W E R T = vowels (steer the held engine).
//! Bow surface: hold, x-offset from center = speed, height = pressure.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

use timber::util::{AtomicF32, Rng, SR};
use timber::voice::{self, Note, Vowel};
use timber::{body, cymbal, drums, mesh, modal, mouth, sing, speak, stream, tract};

// ---- Lock-free mixer state shared with the audio thread -----------------

struct Strip {
    gain: AtomicF32,
    pan: AtomicF32, // -1..1
    mute: AtomicBool,
    /// Index into body::PRESETS.
    body: AtomicUsize,
    body_wet: AtomicF32,
}

struct Mixer {
    strips: [Strip; 3], // drums, string, voice
    master: AtomicF32,
}

const DRUMS: usize = 0;
const STRING: usize = 1;
const VOICE: usize = 2;
const STRIP_NAMES: [&str; 3] = ["drums", "string", "voice"];

/// Points in the string-motion scope the audio thread keeps refreshed.
const SCOPE_LEN: usize = 128;

enum Msg {
    /// A finished render for the buffer player (voice one-shots).
    Buffer(usize, Vec<f32>, Option<u8>),
    /// Pluck one streaming string.
    Pluck(usize),
    /// Hit one streaming drum pad.
    Strike(usize),
    /// A pad's parameters changed — the ringing state carries on under
    /// the new settings.
    Pad(usize, drums::PadParams),
    /// Rap a knuckle on a strip's body: an impulse through the body
    /// filter alone, to hear the box itself.
    Knock(usize),
    /// Hold-to-roll on a drum pad: on while the key is held.
    Roll(usize, bool),
    /// Speak a phoneme sequence through the tract.
    Speak(Vec<speak::Seg>),
    /// Hit a finite-difference membrane pad with a stick velocity.
    MeshStrike(usize, f32),
    /// A mesh pad's parameters changed.
    MeshPad(usize, mesh::MeshParams),
    /// Hit a cymbal with a stick velocity; change its params.
    CymbalStrike(usize, f32),
    CymbalPad(usize, cymbal::CymbalParams),
}

// ---- Audio thread --------------------------------------------------------

struct PlayVoice {
    buf: Vec<f32>,
    pos: f32,
    strip: usize,
    /// Voices sharing a choke group silence each other: a new hat hit
    /// stops the ringing open hat, the way one physical instrument would.
    choke: Option<u8>,
    /// 1.0 while alive; a choked voice ramps this to 0 over a few ms
    /// (cutting instantly would click) and is then dropped.
    fade: f32,
    dying: bool,
}

fn start_audio(
    mixer: Arc<Mixer>,
    ctl: Arc<stream::Ctl>,
    mouth_ctl: Arc<mouth::Ctl>,
    tract_ctl: Arc<tract::Ctl>,
    scope: Arc<Mutex<Vec<f32>>>,
    rx: Receiver<Msg>,
    string_freqs: Vec<f32>,
    pads: Vec<drums::PadParams>,
    mesh_pads: Vec<mesh::MeshParams>,
    cymbal_pads: Vec<cymbal::CymbalParams>,
    modes: Arc<cymbal::Modes>,
) -> cpal::Stream {
    let device = cpal::default_host()
        .default_output_device()
        .expect("no audio output device");
    let config = device.default_output_config().expect("no output config");
    assert!(
        config.sample_format() == cpal::SampleFormat::F32,
        "unsupported sample format {:?}",
        config.sample_format()
    );
    let config: cpal::StreamConfig = config.into();
    let channels = config.channels as usize;
    // Buffers render at 44.1k; if the device runs at another rate we just
    // read them at a fractional step (linear interpolation). The string
    // bank ticks per output frame, so its pitch would shift on such a
    // device — acceptable for now.
    let step = SR / config.sample_rate.0 as f32;
    let choke_step = 1.0 / (0.004 * config.sample_rate.0 as f32);

    let mut bank = stream::Bank::new(&string_freqs);
    let mut kit = drums::Kit::new(pads);
    let mut heads = mesh::Kit::new(mesh_pads);
    let mut cymbals = cymbal::Kit::new(modes, cymbal_pads);
    let mut mouth = mouth::Mouth::new();
    let mut tube = tract::Tract::new();
    let mut utter: Option<speak::Utterance> = None;
    // One body instance per (strip, preset); the strip's atomic picks
    // which one the signal passes through.
    let mut bodies: Vec<Vec<body::Body>> = (0..3)
        .map(|_| {
            body::PRESETS
                .iter()
                .map(|(_, m, ringy)| body::Body::new(m, *ringy))
                .collect()
        })
        .collect();
    let mut rng = Rng(0x626f7765);
    let mut voices: Vec<PlayVoice> = Vec::new();
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let p = stream::Params::read(&ctl);
                let mp = mouth::Params::read(&mouth_ctl);
                let tp = tract::Params::read(&tract_ctl);
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        Msg::Pluck(i) => bank.pluck(i, ctl.pluck_pos.get(), &mut rng),
                        Msg::Strike(i) => kit.strike(i),
                        Msg::Pad(i, params) => kit.set_params(i, params),
                        Msg::Roll(i, on) => kit.set_roll(i, on),
                        Msg::Speak(segs) => utter = speak::Utterance::new(segs),
                        Msg::MeshStrike(i, strength) => heads.strike(i, strength),
                        Msg::MeshPad(i, params) => heads.set_params(i, params),
                        Msg::CymbalStrike(i, strength) => cymbals.strike(i, strength),
                        Msg::CymbalPad(i, params) => cymbals.set_params(i, params),
                        Msg::Knock(i) => {
                            let sel = mixer.strips[i].body.load(Relaxed).min(bodies[i].len() - 1);
                            if sel > 0 {
                                bodies[i][sel].knock(0.8);
                            }
                        }
                        Msg::Buffer(strip, buf, choke) => {
                            if let Some(group) = choke {
                                for v in voices.iter_mut() {
                                    if v.choke == Some(group) {
                                        v.dying = true;
                                    }
                                }
                            }
                            // At the soft cap, retire the oldest voice
                            // with the same quick fade a choke gets;
                            // hard-drop only if truly flooded.
                            if voices.len() >= 96 {
                                voices.remove(0);
                            }
                            if voices.len() >= 64
                                && let Some(v) = voices.iter_mut().find(|v| !v.dying)
                            {
                                v.dying = true;
                            }
                            voices.push(PlayVoice {
                                buf,
                                pos: 0.0,
                                strip,
                                choke,
                                fade: 1.0,
                                dying: false,
                            });
                        }
                    }
                }
                for frame in data.chunks_mut(channels) {
                    // Sum each strip dry first: the body must see the
                    // strip's whole signal once, not each voice.
                    let mut sums = [0.0f32; 3];
                    voices.retain_mut(|v| {
                        let i = v.pos as usize;
                        if i + 1 >= v.buf.len() {
                            return false;
                        }
                        let frac = v.pos - i as f32;
                        let s = (v.buf[i] * (1.0 - frac) + v.buf[i + 1] * frac) * v.fade;
                        v.pos += step;
                        if v.dying {
                            v.fade -= choke_step;
                            if v.fade <= 0.0 {
                                return false;
                            }
                        }
                        sums[v.strip] += s;
                        true
                    });
                    sums[STRING] += bank.tick(&p);
                    sums[DRUMS] +=
                        kit.tick(&mut rng) + heads.tick(&mut rng) + cymbals.tick(&mut rng);
                    sums[VOICE] += mouth.tick(&mp, &mut rng);
                    // A running utterance overrides the pad's
                    // articulation; pitch/level stay live. Release
                    // semantics live in the engine (articulation holds
                    // while the gate is off), so the fallback is clean.
                    let p_use = match utter.as_mut().and_then(|u| u.step(&tp)) {
                        Some(sp) => sp,
                        None => {
                            utter = None;
                            tp
                        }
                    };
                    tract_ctl.speaking.store(utter.is_some(), Relaxed);
                    sums[VOICE] += tube.tick(&p_use, &mut rng);

                    let (mut l, mut r) = (0.0f32, 0.0f32);
                    for (i, strip) in mixer.strips.iter().enumerate() {
                        let mut s = sums[i];
                        let sel = strip.body.load(Relaxed).min(bodies[i].len() - 1);
                        if sel > 0 {
                            // Dry↔body crossfade. Full wet is the
                            // physically honest case: a bare string
                            // moves almost no air — in reality you only
                            // ever hear the box.
                            let wet = strip.body_wet.get();
                            let boxed = bodies[i][sel].tick(sums[i]);
                            s = s * (1.0 - wet) + boxed * wet;
                        }
                        if !strip.mute.load(Relaxed) {
                            // Equal-power pan.
                            let a = (strip.pan.get() + 1.0) * std::f32::consts::FRAC_PI_4;
                            let g = strip.gain.get() * s;
                            l += g * a.cos();
                            r += g * a.sin();
                        }
                    }
                    let m = mixer.master.get();
                    frame[0] = (l * m).tanh();
                    if channels > 1 {
                        frame[1] = (r * m).tanh();
                    }
                    for c in frame.iter_mut().skip(2) {
                        *c = 0.0;
                    }
                }
                if let Ok(mut s) = scope.try_lock() {
                    bank.shape(p.bow_string, &mut s);
                }
            },
            |e| eprintln!("audio error: {e}"),
            None,
        )
        .expect("failed to build audio stream");
    stream.play().expect("failed to start audio stream");
    stream
}

// ---- Instrument state ----------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ModeSet {
    Membrane,
    Center,
    Bell,
    Triangle,
    Cymbal,
    Ride,
    NoiseOnly,
}

impl ModeSet {
    fn slice(self) -> &'static [modal::Mode] {
        match self {
            ModeSet::Membrane => modal::MEMBRANE,
            ModeSet::Center => modal::MEMBRANE_CENTER,
            ModeSet::Bell => modal::BELL,
            ModeSet::Triangle => modal::TRIANGLE,
            ModeSet::Cymbal => modal::CYMBAL,
            ModeSet::Ride => modal::RIDE,
            ModeSet::NoiseOnly => &[],
        }
    }
    fn name(self) -> &'static str {
        match self {
            ModeSet::Membrane => "membrane",
            ModeSet::Center => "membrane (center hit)",
            ModeSet::Bell => "bell",
            ModeSet::Triangle => "triangle (rod)",
            ModeSet::Cymbal => "cymbal (synth)",
            ModeSet::Ride => "ride (synth)",
            ModeSet::NoiseOnly => "noise only",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct DrumParams {
    name: &'static str,
    modes: ModeSet,
    freq: f32,
    glide: f32,
    glide_time: f32,
    damp: f32,
    noise: f32,
    noise_decay: f32,
    noise_tone: f32,
    bloom: f32,
    drive: f32,
    shimmer: f32,
    level: f32,
    choke: Option<u8>,
}

impl DrumParams {
    fn to_pad(self) -> drums::PadParams {
        drums::PadParams {
            modes: self.modes.slice(),
            freq: self.freq,
            glide: self.glide,
            glide_time: self.glide_time,
            damp: self.damp,
            noise: self.noise,
            noise_decay: self.noise_decay,
            noise_tone: self.noise_tone,
            bloom: self.bloom,
            drive: self.drive,
            shimmer: self.shimmer,
            level: self.level,
            choke: self.choke,
        }
    }
}

fn default_pads() -> Vec<DrumParams> {
    let base = DrumParams {
        name: "",
        modes: ModeSet::Membrane,
        freq: 110.0,
        glide: 0.0,
        glide_time: 0.1,
        damp: 1.0,
        noise: 0.0,
        noise_decay: 0.1,
        noise_tone: 0.0,
        bloom: 0.0,
        drive: 0.0,
        shimmer: 0.0,
        level: 0.85,
        choke: None,
    };
    vec![
        // Tight studio kick: tuned up where ears respond, heavily
        // muffled, small fast pitch settle, beater click, some drive.
        DrumParams {
            name: "kick",
            modes: ModeSet::Center,
            freq: 60.0,
            glide: 0.5,
            glide_time: 0.035,
            damp: 0.35,
            // A touch of very fast rattle is the beater's impact click.
            noise: 0.18,
            noise_decay: 0.004,
            drive: 1.8,
            level: 1.0,
            ..base
        },
        DrumParams {
            name: "snare",
            freq: 185.0,
            glide: 0.15,
            glide_time: 0.05,
            noise: 0.9,
            noise_decay: 0.09,
            ..base
        },
        DrumParams {
            name: "tom hi",
            freq: 155.0,
            glide: 0.25,
            glide_time: 0.12,
            ..base
        },
        DrumParams {
            name: "tom",
            freq: 110.0,
            glide: 0.25,
            glide_time: 0.12,
            ..base
        },
        DrumParams {
            name: "tom lo",
            freq: 80.0,
            glide: 0.25,
            glide_time: 0.12,
            ..base
        },
        DrumParams {
            name: "hat",
            modes: ModeSet::NoiseOnly,
            noise: 1.0,
            noise_decay: 0.025,
            level: 0.4,
            choke: Some(0),
            ..base
        },
        DrumParams {
            name: "open hat",
            modes: ModeSet::NoiseOnly,
            noise: 1.0,
            noise_decay: 0.18,
            level: 0.4,
            choke: Some(0),
            ..base
        },
        DrumParams {
            name: "bell",
            modes: ModeSet::Bell,
            freq: 440.0,
            level: 0.6,
            ..base
        },
        // The strike's contact click is a whisper of very fast rattle.
        DrumParams {
            name: "triangle",
            modes: ModeSet::Triangle,
            freq: 1180.0,
            noise: 0.04,
            noise_decay: 0.002,
            level: 0.5,
            ..base
        },
        // Synth cymbals: the CYMBAL partial table with the pad's noise
        // burst as the wash. The crash is mostly wash, the ride mostly
        // partials with a short sizzle.
        DrumParams {
            name: "crash",
            modes: ModeSet::Cymbal,
            freq: 450.0,
            damp: 1.0,
            noise: 0.6,
            noise_decay: 0.7,
            noise_tone: 1.0,
            bloom: 0.8,
            drive: 0.6,
            shimmer: 0.4,
            level: 0.45,
            ..base
        },
        DrumParams {
            name: "ride",
            modes: ModeSet::Ride,
            freq: 340.0,
            damp: 2.0,
            noise: 0.4,
            noise_decay: 0.15,
            noise_tone: 0.8,
            bloom: 0.3,
            // High: the ride's partials ring for seconds, and a slow
            // beat on a long partial is a wah, not a shimmer.
            shimmer: 1.0,
            level: 0.45,
            ..base
        },
    ]
}

const VOWELS: [(&str, Vowel); 5] = [
    ("ah", voice::AH),
    ("ee", voice::EE),
    ("oo", voice::OO),
    ("eh", voice::EH),
    ("oh", voice::OH),
];

/// Pad keys, matched against the *physical* key when the event carries
/// one (shift can change the logical key on some layouts — QWERTZ turns
/// shift+',' into ';' — which would strand roll state). Must stay in
/// step with default_pads(): asserted at startup.
const DRUM_KEYS: [egui::Key; 11] = [
    egui::Key::Z,
    egui::Key::X,
    egui::Key::C,
    egui::Key::V,
    egui::Key::B,
    egui::Key::N,
    egui::Key::M,
    egui::Key::Comma,
    egui::Key::Period,
    egui::Key::Num8,
    egui::Key::Num9,
];
const NPADS: usize = DRUM_KEYS.len();
/// Mesh pads: number row. Shift = a hard hit (the tension glide and
/// the snare's full rattle only show up when the stick really lands).
const MESH_KEYS: [egui::Key; 5] = [
    egui::Key::Num1,
    egui::Key::Num2,
    egui::Key::Num3,
    egui::Key::Num4,
    egui::Key::Num5,
];
const NMESH: usize = MESH_KEYS.len();
/// Cymbal pads: 6 7. Shift = hard hit.
const CYMBAL_KEYS: [egui::Key; 2] = [egui::Key::Num6, egui::Key::Num7];
const NCYMBAL: usize = CYMBAL_KEYS.len();
const NOTE_KEYS: [egui::Key; 8] = [
    egui::Key::A,
    egui::Key::S,
    egui::Key::D,
    egui::Key::F,
    egui::Key::G,
    egui::Key::H,
    egui::Key::J,
    egui::Key::K,
];
const VOWEL_KEYS: [egui::Key; 5] = [
    egui::Key::Q,
    egui::Key::W,
    egui::Key::E,
    egui::Key::R,
    egui::Key::T,
];

// A-minor pentatonic across two octaves, matching the demo.
const NOTES: [(&str, f32); 8] = [
    ("A2", 110.0),
    ("C3", 130.81),
    ("D3", 146.83),
    ("E3", 164.81),
    ("G3", 196.0),
    ("A3", 220.0),
    ("C4", 261.63),
    ("D4", 293.66),
];

// ---- The app -------------------------------------------------------------

struct Desk {
    mixer: Arc<Mixer>,
    ctl: Arc<stream::Ctl>,
    mouth: Arc<mouth::Ctl>,
    tract: Arc<tract::Ctl>,
    tract_mode: bool,
    phrase: String,
    scope: Arc<Mutex<Vec<f32>>>,
    tx: Sender<Msg>,
    _stream: cpal::Stream,
    rng: Rng,

    pads: Vec<DrumParams>,
    pad_sel: usize,
    mesh_pads: Vec<(&'static str, mesh::MeshParams)>,
    mesh_sel: usize,
    cymbal_pads: Vec<(&'static str, cymbal::CymbalParams)>,
    cymbal_sel: usize,
    rolling: [bool; NPADS],
    drum_down: [bool; NPADS],
    last_mouth_pos: Option<egui::Pos2>,

    note: Note,
    from_vowel: usize,
    to_vowel: usize,
    voice_freq: f32,
}

impl Desk {
    fn strike(&mut self, pad: usize) {
        self.pad_sel = pad;
        let _ = self.tx.send(Msg::Strike(pad));
    }

    fn strike_mesh(&mut self, pad: usize, hard: bool) {
        self.mesh_sel = pad;
        let strength = if hard { 1.6 } else { 0.8 };
        let _ = self.tx.send(Msg::MeshStrike(pad, strength));
    }

    fn strike_cymbal(&mut self, pad: usize, hard: bool) {
        self.cymbal_sel = pad;
        let strength = if hard { 1.6 } else { 0.8 };
        let _ = self.tx.send(Msg::CymbalStrike(pad, strength));
    }

    fn pluck(&mut self, string: usize) {
        let _ = self.tx.send(Msg::Pluck(string));
        self.ctl.bow_string.store(string, Relaxed);
    }

    fn sing(&mut self, from: usize, to: usize) {
        let n = Note {
            freq: self.voice_freq,
            from: VOWELS[from].1,
            to: VOWELS[to].1,
            ..self.note
        };
        let _ = self
            .tx
            .send(Msg::Buffer(VOICE, voice::render(&n, &mut self.rng), None));
    }
}

/// Track width that leaves room for the value box and the widest label
/// ("rattle decay" ≈ 82 + 48 + 16) without spilling into the next column.
fn track_width(ui: &egui::Ui) -> f32 {
    (ui.available_width() - 170.0).clamp(40.0, 200.0)
}

fn slider(
    ui: &mut egui::Ui,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    log: bool,
    label: &str,
) -> bool {
    ui.spacing_mut().slider_width = track_width(ui);
    ui.add(egui::Slider::new(v, range).logarithmic(log).text(label))
        .changed()
}

/// A slider over a thread-shared parameter: the audio side sees the move
/// immediately, on notes already sounding.
fn ctl_slider(
    ui: &mut egui::Ui,
    at: &AtomicF32,
    range: std::ops::RangeInclusive<f32>,
    log: bool,
    label: &str,
) {
    let mut v = at.get();
    ui.spacing_mut().slider_width = track_width(ui);
    if ui
        .add(
            egui::Slider::new(&mut v, range)
                .logarithmic(log)
                .text(label),
        )
        .changed()
    {
        at.set(v);
    }
}

impl eframe::App for Desk {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keep repainting so the string scope animates while notes ring.
        ctx.request_repaint_after(std::time::Duration::from_millis(33));

        // ---- Keyboard: collect first, act after, to keep borrows simple.
        // Real key presses only (OS key-repeat would machine-gun hits),
        // matched on the physical key where available so shift and
        // layout can't reroute a held pad key.
        let mut acts: Vec<(usize, usize)> = Vec::new(); // (kind, index)
        let typing = ctx.wants_keyboard_input();
        ctx.input(|i| {
            if typing {
                // A focused text field owns the keyboard: no plucks,
                // strikes or rolls while spelling out phonemes.
                self.drum_down = [false; NPADS];
                for k in 0..NPADS {
                    if self.rolling[k] {
                        self.rolling[k] = false;
                        let _ = self.tx.send(Msg::Roll(k, false));
                    }
                }
                return;
            }
            for ev in &i.events {
                let egui::Event::Key {
                    key,
                    physical_key,
                    pressed,
                    repeat,
                    ..
                } = ev
                else {
                    continue;
                };
                let key = physical_key.unwrap_or(*key);
                if let Some(k) = DRUM_KEYS.iter().position(|d| *d == key) {
                    self.drum_down[k] = *pressed;
                    if *pressed && !*repeat {
                        acts.push((0, k));
                    }
                } else if *pressed && !*repeat {
                    if let Some(k) = MESH_KEYS.iter().position(|d| *d == key) {
                        acts.push((if i.modifiers.shift { 4 } else { 3 }, k));
                    } else if let Some(k) = CYMBAL_KEYS.iter().position(|d| *d == key) {
                        acts.push((if i.modifiers.shift { 6 } else { 5 }, k));
                    } else if let Some(k) = NOTE_KEYS.iter().position(|d| *d == key) {
                        acts.push((1, k));
                    } else if let Some(k) = VOWEL_KEYS.iter().position(|d| *d == key) {
                        acts.push((2, k));
                    }
                }
            }
            // Shift+drum-key held = roll; a plain press stays a clean
            // single hit.
            for k in 0..NPADS {
                let down = self.drum_down[k] && i.modifiers.shift;
                if down != self.rolling[k] {
                    self.rolling[k] = down;
                    let _ = self.tx.send(Msg::Roll(k, down));
                }
            }
        });
        for (kind, k) in acts {
            match kind {
                0 => self.strike(k),
                3 => self.strike_mesh(k, false),
                4 => self.strike_mesh(k, true),
                5 => self.strike_cymbal(k, false),
                6 => self.strike_cymbal(k, true),
                // Left hand fingers, right hand excites: while the bow
                // is on the string, a key only changes the fingered
                // note under the sustained stroke — no pluck. With the
                // bow lifted, keys pluck as before.
                1 if self.ctl.bow_on.load(Relaxed) => {
                    self.ctl.bow_string.store(k, Relaxed);
                }
                // While a voice engine is held, note keys re-pitch it
                // (the surface hand steers vowels, this hand melody).
                1 if self.tract.gate.load(Relaxed) || self.tract.speaking.load(Relaxed) => {
                    self.voice_freq = NOTES[k].1
                }
                1 if self.mouth.gate.load(Relaxed) => self.voice_freq = NOTES[k].1,
                1 => self.pluck(k),
                // While a voice engine is held, vowel keys steer it —
                // snap to that vowel's articulation (the smoothing
                // glides there) instead of layering a one-shot voice.
                _ if self.tract.gate.load(Relaxed) => {
                    let (_, tp, con, lips) = tract::VOWELS[k];
                    self.tract.tongue_pos.set(tp);
                    self.tract.constrict.set(con);
                    self.tract.lips.set(lips);
                }
                _ if self.mouth.gate.load(Relaxed) => {
                    let v = &VOWELS[k].1;
                    self.mouth.f1.set(v.0[0].0);
                    self.mouth.f2.set(v.0[1].0);
                }
                _ => self.sing(k, k),
            }
        }

        // ---- Mixer strips on the right.
        egui::SidePanel::right("mixer").show(ctx, |ui| {
            ui.heading("mixer");
            ui.horizontal(|ui| {
                for (i, strip) in self.mixer.strips.iter().enumerate() {
                    ui.vertical(|ui| {
                        ui.label(STRIP_NAMES[i]);
                        let mut g = strip.gain.get();
                        if ui
                            .add(egui::Slider::new(&mut g, 0.0..=1.2).vertical())
                            .changed()
                        {
                            strip.gain.set(g);
                        }
                        let mut p = strip.pan.get();
                        if ui
                            .add(egui::Slider::new(&mut p, -1.0..=1.0).show_value(false))
                            .changed()
                        {
                            strip.pan.set(p);
                        }
                        let mut m = strip.mute.load(Relaxed);
                        if ui.toggle_value(&mut m, "mute").changed() {
                            strip.mute.store(m, Relaxed);
                        }
                    });
                }
                ui.vertical(|ui| {
                    ui.label("master");
                    let mut g = self.mixer.master.get();
                    if ui
                        .add(egui::Slider::new(&mut g, 0.0..=1.5).vertical())
                        .changed()
                    {
                        self.mixer.master.set(g);
                    }
                });
            });
            ui.separator();
            ui.label("bodies");
            ui.small("slider: dry ← → body");
            for (i, strip) in self.mixer.strips.iter().enumerate() {
                ui.horizontal(|ui| {
                    let mut sel = strip.body.load(Relaxed).min(body::PRESETS.len() - 1);
                    egui::ComboBox::from_id_salt(("body", i))
                        .width(70.0)
                        .selected_text(body::PRESETS[sel].0)
                        .show_ui(ui, |ui| {
                            for (k, (name, _, _)) in body::PRESETS.iter().enumerate() {
                                ui.selectable_value(&mut sel, k, *name);
                            }
                        });
                    strip.body.store(sel, Relaxed);
                    let mut wet = strip.body_wet.get();
                    if ui
                        .add(
                            egui::Slider::new(&mut wet, 0.0..=1.0)
                                .show_value(false)
                                .text(STRIP_NAMES[i]),
                        )
                        .changed()
                    {
                        strip.body_wet.set(wet);
                    }
                    if ui.small_button("knock").clicked() {
                        let _ = self.tx.send(Msg::Knock(i));
                    }
                });
            }
            ui.separator();
            ui.small("Z X C V B N M , . 8 9 — drums (shift = roll)");
            ui.small("1 2 3 4 5 — mesh drums (shift = hard hit)");
            ui.small("6 7 — cymbals (shift = hard hit)");
            ui.small("A S D F G H J K — pluck (finger, while bowing)");
            ui.small("Q W E R T — vowels");
            ui.small("hold bow surface — bow");
            ui.small("hold mouth surface — sing");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Three kits and a voice no longer fit a laptop screen.
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(3, |cols| {
                    // ---- Drums.
                    let ui = &mut cols[0];
                    ui.heading("drums · modal");
                    let mut strikes = Vec::new();
                    ui.horizontal_wrapped(|ui| {
                        for (i, p) in self.pads.iter().enumerate() {
                            if ui.selectable_label(self.pad_sel == i, p.name).clicked() {
                                strikes.push(i);
                            }
                        }
                    });
                    for i in strikes {
                        self.strike(i);
                    }
                    ui.separator();
                    let p = &mut self.pads[self.pad_sel];
                    // Response-based edit detection: comparing the struct
                    // before/after would mistake egui's clamp-on-show for a
                    // user edit (the silent-retune bug).
                    let mut edited = false;
                    ui.label(format!("editing: {}", p.name));
                    egui::ComboBox::from_label("object")
                        .width((ui.available_width() - 130.0).clamp(80.0, 160.0))
                        .selected_text(p.modes.name())
                        .show_ui(ui, |ui| {
                            for m in [
                                ModeSet::Membrane,
                                ModeSet::Center,
                                ModeSet::Bell,
                                ModeSet::Triangle,
                                ModeSet::Cymbal,
                                ModeSet::Ride,
                                ModeSet::NoiseOnly,
                            ] {
                                edited |= ui.selectable_value(&mut p.modes, m, m.name()).changed();
                            }
                        });
                    // Range must still contain every pad's default: egui
                    // clamps an out-of-range value on display, silently
                    // mutating the stored param (shipped on the next real
                    // edit). The Response-based `edited` flag keeps the
                    // clamp itself from *triggering* a send — the original
                    // "mwup" bug — but not from corrupting the value.
                    edited |= slider(ui, &mut p.freq, 25.0..=2400.0, true, "freq");
                    edited |= slider(ui, &mut p.glide, 0.0..=1.5, false, "pitch glide");
                    edited |= slider(ui, &mut p.glide_time, 0.01..=0.5, true, "glide time");
                    // Floor low enough to be a hand grabbing the metal: at
                    // 0.02 a six-second triangle mode dies in ~0.1s.
                    edited |= slider(ui, &mut p.damp, 0.02..=3.0, true, "muffle");
                    edited |= slider(ui, &mut p.noise, 0.0..=1.0, false, "rattle");
                    edited |= slider(ui, &mut p.noise_decay, 0.002..=1.5, true, "rattle decay");
                    edited |= slider(ui, &mut p.noise_tone, 0.0..=1.0, false, "rattle tone");
                    edited |= slider(ui, &mut p.bloom, 0.0..=1.0, false, "wash bloom");
                    edited |= slider(ui, &mut p.drive, 0.0..=6.0, false, "drive");
                    edited |= slider(ui, &mut p.shimmer, 0.0..=1.0, false, "shimmer");
                    edited |= slider(ui, &mut p.level, 0.0..=1.0, false, "level");
                    let mut chokes = p.choke.is_some();
                    if ui.checkbox(&mut chokes, "choke group").changed() {
                        p.choke = if chokes { Some(0) } else { None };
                        edited = true;
                    }
                    // Ship changed params to the ringing pad — knob moves
                    // land on sounds already in the air.
                    if edited {
                        let _ = self.tx.send(Msg::Pad(self.pad_sel, p.to_pad()));
                    }

                    // ---- Mesh drums: the membrane as a membrane.
                    ui.separator();
                    ui.heading("drums · mesh");
                    let mut hits = Vec::new();
                    ui.horizontal_wrapped(|ui| {
                        for (i, (name, _)) in self.mesh_pads.iter().enumerate() {
                            let r = ui.selectable_label(self.mesh_sel == i, *name);
                            if r.clicked() {
                                hits.push((i, ui.input(|inp| inp.modifiers.shift)));
                            }
                        }
                    });
                    for (i, hard) in hits {
                        self.strike_mesh(i, hard);
                    }
                    let (name, m) = &mut self.mesh_pads[self.mesh_sel];
                    let mut edited = false;
                    ui.label(format!("editing: {name}"));
                    edited |= slider(ui, &mut m.freq, 30.0..=400.0, true, "freq");
                    edited |= slider(ui, &mut m.decay, 0.05..=3.0, true, "decay");
                    edited |= slider(ui, &mut m.hf_damp, 0.0..=0.95, false, "overtone damp");
                    edited |= slider(ui, &mut m.tension, 0.0..=80.0, false, "tension");
                    edited |= slider(ui, &mut m.strike_pos, 0.0..=1.0, false, "strike pos");
                    edited |= slider(ui, &mut m.rattle, 0.0..=1.0, false, "rattle");
                    edited |= slider(ui, &mut m.click, 0.0..=1.0, false, "beater click");
                    edited |= slider(ui, &mut m.drive, 0.0..=6.0, false, "drive");
                    edited |= slider(ui, &mut m.air, 0.0..=1.5, false, "shell air");
                    edited |= slider(ui, &mut m.reso_freq, 25.0..=300.0, true, "reso head");
                    edited |= slider(ui, &mut m.reso_decay, 0.05..=3.0, true, "reso decay");
                    edited |= slider(ui, &mut m.hardness, 0.0..=1.0, false, "stick hardness");
                    edited |= slider(ui, &mut m.mallet, 0.8..=5.0, false, "mallet size");
                    edited |= slider(ui, &mut m.level, 0.0..=4.0, false, "level");
                    if edited {
                        let _ = self.tx.send(Msg::MeshPad(self.mesh_sel, *m));
                    }

                    // ---- Strings.
                    let ui = &mut cols[1];
                    ui.heading("string · streaming");
                    let mut plucks = Vec::new();
                    ui.horizontal_wrapped(|ui| {
                        for (i, (name, _)) in NOTES.iter().enumerate() {
                            if ui.button(*name).clicked() {
                                plucks.push(i);
                            }
                        }
                    });
                    for i in plucks {
                        self.pluck(i);
                    }
                    ui.separator();
                    ctl_slider(ui, &self.ctl.decay, 0.95..=0.9995, false, "decay");
                    ctl_slider(ui, &self.ctl.damping, 0.0..=1.0, false, "damping");
                    ctl_slider(ui, &self.ctl.pluck_pos, 0.0..=0.5, false, "pluck pos");
                    ctl_slider(ui, &self.ctl.stiffness, 0.0..=0.9, false, "stiffness");
                    ctl_slider(ui, &self.ctl.couple, 0.0..=0.01, false, "sympathy");
                    ctl_slider(ui, &self.ctl.level, 0.0..=1.0, false, "level");
                    ui.separator();

                    // ---- The bow: hold on the surface. Horizontal *position*
                    // is bow speed — right of center bows forward, left bows
                    // back, the middle rests the bow on the string (which
                    // damps it, as a real resting bow does). Height is
                    // pressure. Position instead of drag velocity because a
                    // trackpad is 10cm of glass standing in for 70cm of bow
                    // hair: sustained strokes must not require sustained
                    // motion, and a bow *change* should be a deliberate
                    // crossing of the center, not every scrub reversal.
                    let bowed = self.ctl.bow_string.load(Relaxed);
                    let mut drone = self.ctl.drone.load(Relaxed);
                    if ui
                        .checkbox(&mut drone, "drone — bow all strings at once")
                        .changed()
                    {
                        self.ctl.drone.store(drone, Relaxed);
                    }
                    ui.label(if drone {
                        "bow — all strings (hold: ⇄ = speed, height = pressure)".to_string()
                    } else {
                        format!(
                            "bow — {} (hold: ⇄ from center = speed, height = pressure)",
                            NOTES[bowed].0
                        )
                    });
                    let h = 80.0;
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), h),
                        egui::Sense::drag(),
                    );
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
                    if let Ok(shape) = self.scope.lock() {
                        let pts: Vec<egui::Pos2> = shape
                            .iter()
                            .enumerate()
                            .map(|(i, s)| {
                                egui::pos2(
                                    rect.left()
                                        + rect.width() * i as f32 / (shape.len() - 1) as f32,
                                    rect.center().y - s.clamp(-1.2, 1.2) * (h * 0.42),
                                )
                            })
                            .collect();
                        painter.add(egui::Shape::line(
                            pts,
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(130, 190, 255)),
                        ));
                    }
                    // Center line: the bow at rest.
                    painter.vline(
                        rect.center().x,
                        rect.y_range(),
                        egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
                    );
                    let held = resp.dragged() || resp.is_pointer_button_down_on();
                    let pos = resp.interact_pointer_pos();
                    if held && let Some(pos) = pos {
                        let x = ((pos.x - rect.center().x) / (rect.width() * 0.5)).clamp(-1.0, 1.0);
                        // Small dead zone: resting the bow, not moving it.
                        let speed = if x.abs() < 0.06 { 0.0 } else { x * 1.1 };
                        let pressure = ((pos.y - rect.top()) / h).clamp(0.05, 1.0);
                        painter.circle_filled(
                            pos.clamp(rect.min, rect.max),
                            4.0,
                            egui::Color32::from_rgb(255, 180, 90),
                        );
                        self.ctl.bow_speed.set(speed);
                        self.ctl.bow_pressure.set(pressure);
                        self.ctl.bow_on.store(true, Relaxed);
                    } else {
                        self.ctl.bow_on.store(false, Relaxed);
                    }

                    // ---- Cymbals: the modal plate.
                    ui.separator();
                    ui.heading("cymbals · modal plate");
                    let mut hits = Vec::new();
                    ui.horizontal_wrapped(|ui| {
                        for (i, (name, _)) in self.cymbal_pads.iter().enumerate() {
                            let r = ui.selectable_label(self.cymbal_sel == i, *name);
                            if r.clicked() {
                                hits.push((i, ui.input(|inp| inp.modifiers.shift)));
                            }
                        }
                    });
                    for (i, hard) in hits {
                        self.strike_cymbal(i, hard);
                    }
                    let (name, c) = &mut self.cymbal_pads[self.cymbal_sel];
                    let mut edited = false;
                    ui.label(format!("editing: {name}"));
                    edited |= slider(ui, &mut c.stiffness, 0.0..=1.0, false, "stiffness");
                    edited |= slider(ui, &mut c.dome, 10.0..=400.0, true, "dome");
                    edited |= slider(ui, &mut c.decay, 0.1..=12.0, true, "decay");
                    edited |= slider(ui, &mut c.hf_damp, 0.0..=1.0, false, "hf damp");
                    edited |= slider(ui, &mut c.strike_pos, 0.0..=1.0, false, "strike pos");
                    edited |= slider(ui, &mut c.hardness, 0.0..=1.0, false, "stick hardness");
                    edited |= slider(ui, &mut c.mallet, 0.8..=5.0, false, "mallet size");
                    edited |= slider(ui, &mut c.nonlin, 1.0..=5000.0, true, "wash (nonlin)");
                    edited |= slider(ui, &mut c.level, 0.0..=4.0, false, "level");
                    if edited {
                        let _ = self.tx.send(Msg::CymbalPad(self.cymbal_sel, *c));
                    }

                    // ---- Voice.
                    let ui = &mut cols[2];
                    ui.heading("voice · formant");
                    let sing_now = ui.button("sing").clicked();
                    egui::ComboBox::from_label("from")
                        .selected_text(VOWELS[self.from_vowel].0)
                        .show_ui(ui, |ui| {
                            for (i, (name, _)) in VOWELS.iter().enumerate() {
                                ui.selectable_value(&mut self.from_vowel, i, *name);
                            }
                        });
                    egui::ComboBox::from_label("to")
                        .selected_text(VOWELS[self.to_vowel].0)
                        .show_ui(ui, |ui| {
                            for (i, (name, _)) in VOWELS.iter().enumerate() {
                                ui.selectable_value(&mut self.to_vowel, i, *name);
                            }
                        });
                    slider(ui, &mut self.voice_freq, 70.0..=400.0, true, "pitch");
                    slider(ui, &mut self.note.duration, 0.15..=3.0, true, "duration");
                    slider(
                        ui,
                        &mut self.note.glide_time,
                        0.02..=0.6,
                        true,
                        "vowel glide",
                    );
                    slider(ui, &mut self.note.vibrato, 0.0..=0.03, false, "vibrato");
                    slider(ui, &mut self.note.breath, 0.0..=1.0, false, "breath");
                    slider(ui, &mut self.note.level, 0.0..=1.0, false, "level");
                    if sing_now {
                        self.sing(self.from_vowel, self.to_vowel);
                    }
                    // The streaming mouth shares the sliders' settings.
                    self.mouth.freq.set(self.voice_freq);
                    self.mouth.breath.set(self.note.breath);
                    self.mouth.vibrato.set(self.note.vibrato);
                    self.mouth.level.set(self.note.level);
                    self.tract.freq.set(self.voice_freq);
                    self.tract.breath.set(self.note.breath);
                    self.tract.vibrato.set(self.note.vibrato);
                    self.tract.level.set(self.note.level);
                    ui.separator();
                    let was_tract = self.tract_mode;
                    ui.checkbox(&mut self.tract_mode, "tract engine (Kelly-Lochbaum)");
                    if was_tract != self.tract_mode {
                        self.mouth.gate.store(false, Relaxed);
                        self.tract.gate.store(false, Relaxed);
                        self.last_mouth_pos = None;
                        // An empty utterance stops any running speech — the
                        // hidden engine must not keep talking.
                        let _ = self.tx.send(Msg::Speak(Vec::new()));
                    }
                    if self.tract_mode {
                        // ---- The tract: a tube you sculpt. Drag = tongue
                        // (x along the mouth, y toward closure); the drawn
                        // profile IS the tube the audio thread scatters
                        // through. The pad's range stays below frication
                        // (a tongue can't sustain a gas jet) — consonants
                        // belong to the phoneme box.
                        ctl_slider(ui, &self.tract.lips, 0.0..=1.0, false, "lips");
                        // Nasalized vowels on demand: hold a vowel and open
                        // the nose.
                        ctl_slider(ui, &self.tract.velum, 0.0..=1.0, false, "velum");
                        ui.horizontal(|ui| {
                            if ui.button("speak").clicked() {
                                let segs = speak::phrase(&self.phrase);
                                if !segs.is_empty() {
                                    let _ = self.tx.send(Msg::Speak(segs));
                                }
                            }
                            // The 1961 tribute. The pitch slider transposes.
                            if ui.button("♪ daisy").clicked() {
                                let _ = self.tx.send(Msg::Speak(sing::daisy()));
                            }
                            ui.add(
                                egui::TextEdit::singleline(&mut self.phrase)
                                    .desired_width(ui.available_width()),
                            )
                            .on_hover_text(
                                "phonemes: a e i o u w y h l r s sh f z p b t d k g, '.' pause",
                            );
                        });
                        ui.label("tract — hold to sing: ⇄ tongue back/front, ↓ raise tongue");
                        // The pad maps onto constriction 0..CONSTRICT_MAX:
                        // a tongue can press to frication, but not sustain
                        // the pinhole gas-jet regime beyond it — that zone
                        // is clamped out of the UI, not out of the model.
                        const CONSTRICT_MAX: f32 = 0.87;
                        let h = 80.0;
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), h),
                            egui::Sense::drag(),
                        );
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
                        let d = tract::diameters(
                            self.tract.tongue_pos.get(),
                            self.tract.constrict.get(),
                            self.tract.lips.get(),
                            0.0,
                        );
                        let mut top = Vec::with_capacity(tract::N);
                        let mut bot = Vec::with_capacity(tract::N);
                        for (i, di) in d.iter().enumerate() {
                            let x = rect.left() + rect.width() * i as f32 / (tract::N - 1) as f32;
                            let half = di / 1.6 * (h * 0.42);
                            top.push(egui::pos2(x, rect.center().y - half));
                            bot.push(egui::pos2(x, rect.center().y + half));
                        }
                        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(130, 190, 255));
                        painter.add(egui::Shape::line(top, stroke));
                        painter.add(egui::Shape::line(bot, stroke));
                        for (name, tp, con, _) in tract::VOWELS {
                            painter.text(
                                egui::pos2(
                                    rect.left() + rect.width() * tp,
                                    rect.top() + h * con / CONSTRICT_MAX,
                                ),
                                egui::Align2::CENTER_CENTER,
                                name,
                                egui::FontId::proportional(11.0),
                                ui.visuals().weak_text_color(),
                            );
                        }
                        if (resp.dragged() || resp.is_pointer_button_down_on())
                            && let Some(pos) = resp.interact_pointer_pos()
                        {
                            if self.last_mouth_pos.is_none_or(|p| p.distance(pos) > 1.0) {
                                self.last_mouth_pos = Some(pos);
                                let x = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                                let y = ((pos.y - rect.top()) / h).clamp(0.0, 1.0);
                                self.tract.tongue_pos.set(x);
                                self.tract.constrict.set(y * CONSTRICT_MAX);
                            }
                            self.tract.gate.store(true, Relaxed);
                            let marker = egui::pos2(
                                rect.left() + rect.width() * self.tract.tongue_pos.get(),
                                rect.top() + h * self.tract.constrict.get() / CONSTRICT_MAX,
                            );
                            painter.circle_filled(
                                marker,
                                4.0,
                                egui::Color32::from_rgb(255, 180, 90),
                            );
                        } else {
                            self.tract.gate.store(false, Relaxed);
                            self.last_mouth_pos = None;
                        }
                    } else {
                        // ---- The mouth: hold to phonate, steer through the
                        // phonetician's vowel plane — left/right is tongue
                        // position (F2, front vowels left), up/down is jaw
                        // openness (F1). The named vowels are landmarks in a
                        // continuous space, not the only options.
                        ui.label("mouth — hold to sing, drag between vowels");
                        let h = 80.0;
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), h),
                            egui::Sense::drag(),
                        );
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
                        let (f1_lo, f1_hi) = mouth::F1_RANGE;
                        let (f2_lo, f2_hi) = mouth::F2_RANGE;
                        let to_pos = |f1: f32, f2: f32| {
                            egui::pos2(
                                rect.left()
                                    + rect.width() * (f2_hi / f2).ln() / (f2_hi / f2_lo).ln(),
                                rect.top()
                                    + rect.height() * (f1 / f1_lo).ln() / (f1_hi / f1_lo).ln(),
                            )
                        };
                        for (name, v) in &VOWELS {
                            let (f1, _, _) = v.0[0];
                            let (f2, _, _) = v.0[1];
                            painter.text(
                                to_pos(f1, f2),
                                egui::Align2::CENTER_CENTER,
                                *name,
                                egui::FontId::proportional(12.0),
                                ui.visuals().weak_text_color(),
                            );
                        }
                        if (resp.dragged() || resp.is_pointer_button_down_on())
                            && let Some(pos) = resp.interact_pointer_pos()
                        {
                            // Only a *moving* pointer writes the vowel, so the
                            // Q..T keys can set formants without the resting
                            // finger instantly overwriting them.
                            if self.last_mouth_pos.is_none_or(|p| p.distance(pos) > 1.0) {
                                self.last_mouth_pos = Some(pos);
                                let x = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                                let y = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                                self.mouth.f2.set(f2_hi * (f2_lo / f2_hi).powf(x));
                                self.mouth.f1.set(f1_lo * (f1_hi / f1_lo).powf(y));
                            }
                            self.mouth.gate.store(true, Relaxed);
                            // The marker shows the *actual* formants — pointer
                            // and key agree on one source of truth.
                            let marker = to_pos(self.mouth.f1.get(), self.mouth.f2.get());
                            painter.circle_filled(
                                marker.clamp(rect.min, rect.max),
                                4.0,
                                egui::Color32::from_rgb(255, 180, 90),
                            );
                        } else {
                            self.mouth.gate.store(false, Relaxed);
                            self.last_mouth_pos = None;
                        }
                    }
                });
            });
        });
    }
}

fn main() -> eframe::Result {
    // Bodies are found by name — hardcoded indices silently retarget
    // whenever a preset is inserted (the voice strip once defaulted to
    // cathedral because 'plate' had shifted from 4 to 5).
    let preset = |name: &str| {
        body::PRESETS
            .iter()
            .position(|(n, _, _)| *n == name)
            .expect("unknown body preset")
    };
    let mixer = Arc::new(Mixer {
        strips: [
            Strip {
                gain: AtomicF32::new(0.9),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
                body: AtomicUsize::new(preset("shell")),
                body_wet: AtomicF32::new(0.4),
            },
            Strip {
                gain: AtomicF32::new(0.7),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
                body: AtomicUsize::new(preset("guitar")),
                body_wet: AtomicF32::new(0.5),
            },
            Strip {
                gain: AtomicF32::new(0.8),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
                body: AtomicUsize::new(preset("plate")),
                body_wet: AtomicF32::new(0.3),
            },
        ],
        master: AtomicF32::new(0.8),
    });
    let ctl = Arc::new(stream::Ctl::new());
    let mouth_ctl = Arc::new(mouth::Ctl::new());
    let tract_ctl = Arc::new(tract::Ctl::new());
    let scope = Arc::new(Mutex::new(vec![0.0f32; SCOPE_LEN]));
    let (tx, rx) = channel();
    let freqs = NOTES.iter().map(|(_, f)| *f).collect();
    let pads = default_pads();
    assert_eq!(pads.len(), NPADS, "DRUM_KEYS and default_pads out of step");
    let mesh_pads = mesh::default_kit();
    assert_eq!(
        mesh_pads.len(),
        NMESH,
        "MESH_KEYS and mesh::default_kit out of step"
    );
    // The cymbal's modes and coupling tensor: a second or so of eigen-
    // decomposition, once.
    let t0 = std::time::Instant::now();
    let modes = Arc::new(cymbal::Modes::compute());
    eprintln!(
        "cymbal: {} modes, {} coupling terms, {:.1}s",
        modes.n_modes(),
        modes.coupling_entries(),
        t0.elapsed().as_secs_f32()
    );
    let cymbal_pads = cymbal::default_kit();
    assert_eq!(
        cymbal_pads.len(),
        NCYMBAL,
        "CYMBAL_KEYS and cymbal::default_kit out of step"
    );
    let stream = start_audio(
        mixer.clone(),
        ctl.clone(),
        mouth_ctl.clone(),
        tract_ctl.clone(),
        scope.clone(),
        rx,
        freqs,
        pads.iter().map(|p| p.to_pad()).collect(),
        mesh_pads.iter().map(|(_, m)| *m).collect(),
        cymbal_pads.iter().map(|(_, c)| *c).collect(),
        modes,
    );

    let desk = Desk {
        mixer,
        ctl,
        mouth: mouth_ctl,
        tract: tract_ctl,
        tract_mode: false,
        phrase: "h e l o . w a w".into(),
        scope,
        tx,
        _stream: stream,
        rng: Rng(0x74696d62),
        pads,
        pad_sel: 0,
        mesh_pads,
        mesh_sel: 0,
        cymbal_pads,
        cymbal_sel: 0,
        rolling: [false; NPADS],
        drum_down: [false; NPADS],
        last_mouth_pos: None,
        note: Note::default(),
        from_vowel: 1, // ee → ah: "yah"
        to_vowel: 0,
        voice_freq: 165.0,
    };

    eframe::run_native(
        "timber desk",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1020.0, 760.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(desk))),
    )
}
