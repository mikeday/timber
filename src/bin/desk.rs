//! The timber desk: a little realtime mixing desk over the three models.
//!
//! Architecture: the UI thread holds the model parameters. Hitting a pad
//! renders the note with the exact same offline code the demo uses (these
//! models render orders of magnitude faster than realtime), then ships the
//! buffer to the audio thread, which mixes all live buffers through three
//! channel strips — gain, pan, mute — plus a master with a tanh limiter.
//! Mixer moves are heard live; model knobs take effect on the next trigger.
//!
//! Keys: Z X C V B N = drums · A S D F G H J K = string · Q W E R T = vowels

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender, channel};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

use timber::modal::{self, Hit};
use timber::string::{self, Pluck};
use timber::util::{Rng, SR};
use timber::voice::{self, Note, Vowel};

// ---- Lock-free mixer state shared with the audio thread -----------------

struct AtomicF32(AtomicU32);

impl AtomicF32 {
    fn new(v: f32) -> Self {
        AtomicF32(AtomicU32::new(v.to_bits()))
    }
    fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Relaxed))
    }
    fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Relaxed)
    }
}

struct Strip {
    gain: AtomicF32,
    pan: AtomicF32, // -1..1
    mute: AtomicBool,
}

struct Mixer {
    strips: [Strip; 3], // drums, string, voice
    master: AtomicF32,
}

const DRUMS: usize = 0;
const STRING: usize = 1;
const VOICE: usize = 2;
const STRIP_NAMES: [&str; 3] = ["drums", "string", "voice"];

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

fn start_audio(mixer: Arc<Mixer>, rx: Receiver<(usize, Vec<f32>, Option<u8>)>) -> cpal::Stream {
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
    // Models render at 44.1k; if the device runs at another rate we just
    // read the buffers at a fractional step (linear interpolation).
    let step = SR / config.sample_rate.0 as f32;
    let choke_step = 1.0 / (0.004 * config.sample_rate.0 as f32);

    let mut voices: Vec<PlayVoice> = Vec::new();
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                while let Ok((strip, buf, choke)) = rx.try_recv() {
                    if let Some(group) = choke {
                        for v in voices.iter_mut() {
                            if v.choke == Some(group) {
                                v.dying = true;
                            }
                        }
                    }
                    // At the soft cap, retire the oldest voice with the
                    // same quick fade a choke gets; hard-drop only if
                    // truly flooded.
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
                for frame in data.chunks_mut(channels) {
                    let (mut l, mut r) = (0.0f32, 0.0f32);
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
                        let strip = &mixer.strips[v.strip];
                        if !strip.mute.load(Relaxed) {
                            // Equal-power pan.
                            let a = (strip.pan.get() + 1.0) * std::f32::consts::FRAC_PI_4;
                            let g = strip.gain.get() * s;
                            l += g * a.cos();
                            r += g * a.sin();
                        }
                        true
                    });
                    let m = mixer.master.get();
                    frame[0] = (l * m).tanh();
                    if channels > 1 {
                        frame[1] = (r * m).tanh();
                    }
                    for c in frame.iter_mut().skip(2) {
                        *c = 0.0;
                    }
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
    NoiseOnly,
}

impl ModeSet {
    fn slice(self) -> &'static [modal::Mode] {
        match self {
            ModeSet::Membrane => modal::MEMBRANE,
            ModeSet::Center => modal::MEMBRANE_CENTER,
            ModeSet::Bell => modal::BELL,
            ModeSet::NoiseOnly => &[],
        }
    }
    fn name(self) -> &'static str {
        match self {
            ModeSet::Membrane => "membrane",
            ModeSet::Center => "membrane (center hit)",
            ModeSet::Bell => "bell",
            ModeSet::NoiseOnly => "noise only",
        }
    }
}

#[derive(Clone, Copy)]
struct DrumParams {
    name: &'static str,
    modes: ModeSet,
    freq: f32,
    duration: f32,
    glide: f32,
    glide_time: f32,
    noise: f32,
    noise_decay: f32,
    level: f32,
    choke: Option<u8>,
}

fn default_pads() -> Vec<DrumParams> {
    let base = DrumParams {
        name: "",
        modes: ModeSet::Membrane,
        freq: 110.0,
        duration: 1.2,
        glide: 0.0,
        glide_time: 0.1,
        noise: 0.0,
        noise_decay: 0.1,
        level: 0.85,
        choke: None,
    };
    vec![
        DrumParams {
            name: "kick",
            modes: ModeSet::Center,
            freq: 50.0,
            duration: 0.8,
            glide: 0.9,
            glide_time: 0.06,
            ..base
        },
        DrumParams {
            name: "snare",
            freq: 185.0,
            duration: 0.5,
            glide: 0.15,
            glide_time: 0.05,
            noise: 0.9,
            noise_decay: 0.09,
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
            name: "hat",
            modes: ModeSet::NoiseOnly,
            duration: 0.08,
            noise: 1.0,
            noise_decay: 0.025,
            level: 0.4,
            choke: Some(0),
            ..base
        },
        DrumParams {
            name: "open hat",
            modes: ModeSet::NoiseOnly,
            duration: 0.5,
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
            duration: 5.0,
            level: 0.6,
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
    tx: Sender<(usize, Vec<f32>, Option<u8>)>,
    _stream: cpal::Stream,
    rng: Rng,

    pads: Vec<DrumParams>,
    pad_sel: usize,

    pluck: Pluck,
    note: Note,
    from_vowel: usize,
    to_vowel: usize,
    voice_freq: f32,
}

impl Desk {
    fn strike(&mut self, pad: usize) {
        self.pad_sel = pad;
        let p = self.pads[pad];
        let hit = Hit {
            freq: p.freq,
            modes: p.modes.slice(),
            duration: p.duration,
            glide: p.glide,
            glide_time: p.glide_time.max(0.005),
            noise: p.noise,
            noise_decay: p.noise_decay.max(0.005),
            level: p.level,
        };
        let _ = self
            .tx
            .send((DRUMS, modal::render(&hit, &mut self.rng), p.choke));
    }

    fn pluck_note(&mut self, freq: f32) {
        let p = Pluck { freq, ..self.pluck };
        let _ = self
            .tx
            .send((STRING, string::render(&p, &mut self.rng), None));
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
            .send((VOICE, voice::render(&n, &mut self.rng), None));
    }
}

fn slider(
    ui: &mut egui::Ui,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    log: bool,
    label: &str,
) {
    ui.add(egui::Slider::new(v, range).logarithmic(log).text(label));
}

impl eframe::App for Desk {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ---- Keyboard: collect first, act after, to keep borrows simple.
        use egui::Key;
        const DRUM_KEYS: [Key; 6] = [Key::Z, Key::X, Key::C, Key::V, Key::B, Key::N];
        const NOTE_KEYS: [Key; 8] = [
            Key::A,
            Key::S,
            Key::D,
            Key::F,
            Key::G,
            Key::H,
            Key::J,
            Key::K,
        ];
        const VOWEL_KEYS: [Key; 5] = [Key::Q, Key::W, Key::E, Key::R, Key::T];
        let mut acts: Vec<(usize, usize)> = Vec::new(); // (kind, index)
        ctx.input(|i| {
            for (k, key) in DRUM_KEYS.iter().enumerate() {
                if i.key_pressed(*key) {
                    acts.push((0, k));
                }
            }
            for (k, key) in NOTE_KEYS.iter().enumerate() {
                if i.key_pressed(*key) {
                    acts.push((1, k));
                }
            }
            for (k, key) in VOWEL_KEYS.iter().enumerate() {
                if i.key_pressed(*key) {
                    acts.push((2, k));
                }
            }
        });
        for (kind, k) in acts {
            match kind {
                0 => self.strike(k),
                1 => self.pluck_note(NOTES[k].1),
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
            ui.small("Z X C V B N — drums");
            ui.small("A S D F G H J K — string");
            ui.small("Q W E R T — vowels");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
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
                ui.label(format!("editing: {}", p.name));
                egui::ComboBox::from_label("object")
                    .selected_text(p.modes.name())
                    .show_ui(ui, |ui| {
                        for m in [
                            ModeSet::Membrane,
                            ModeSet::Center,
                            ModeSet::Bell,
                            ModeSet::NoiseOnly,
                        ] {
                            ui.selectable_value(&mut p.modes, m, m.name());
                        }
                    });
                slider(ui, &mut p.freq, 25.0..=880.0, true, "freq");
                slider(ui, &mut p.duration, 0.05..=6.0, true, "duration");
                slider(ui, &mut p.glide, 0.0..=1.5, false, "pitch glide");
                slider(ui, &mut p.glide_time, 0.01..=0.5, true, "glide time");
                slider(ui, &mut p.noise, 0.0..=1.0, false, "rattle");
                slider(ui, &mut p.noise_decay, 0.01..=0.5, true, "rattle decay");
                slider(ui, &mut p.level, 0.0..=1.0, false, "level");
                let mut chokes = p.choke.is_some();
                if ui.checkbox(&mut chokes, "choke group").changed() {
                    p.choke = if chokes { Some(0) } else { None };
                }

                // ---- String.
                let ui = &mut cols[1];
                ui.heading("string · karplus-strong");
                let mut plucks = Vec::new();
                ui.horizontal_wrapped(|ui| {
                    for (name, freq) in NOTES {
                        if ui.button(name).clicked() {
                            plucks.push(freq);
                        }
                    }
                });
                for f in plucks {
                    self.pluck_note(f);
                }
                ui.separator();
                slider(ui, &mut self.pluck.decay, 0.95..=0.9995, false, "decay");
                slider(ui, &mut self.pluck.damping, 0.0..=1.0, false, "damping");
                slider(ui, &mut self.pluck.pluck_pos, 0.0..=0.5, false, "pluck pos");
                slider(ui, &mut self.pluck.stiffness, 0.0..=0.9, false, "stiffness");
                slider(ui, &mut self.pluck.duration, 0.5..=8.0, true, "duration");
                slider(ui, &mut self.pluck.level, 0.0..=1.0, false, "level");

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
            });
        });
    }
}

fn main() -> eframe::Result {
    let mixer = Arc::new(Mixer {
        strips: [
            Strip {
                gain: AtomicF32::new(0.9),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
            },
            Strip {
                gain: AtomicF32::new(0.7),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
            },
            Strip {
                gain: AtomicF32::new(0.8),
                pan: AtomicF32::new(0.0),
                mute: AtomicBool::new(false),
            },
        ],
        master: AtomicF32::new(0.8),
    });
    let (tx, rx) = channel();
    let stream = start_audio(mixer.clone(), rx);

    let desk = Desk {
        mixer,
        tx,
        _stream: stream,
        rng: Rng(0x74696d62),
        pads: default_pads(),
        pad_sel: 0,
        pluck: Pluck {
            level: 0.8,
            ..Default::default()
        },
        note: Note::default(),
        from_vowel: 1, // ee → ah: "yah"
        to_vowel: 0,
        voice_freq: 165.0,
    };

    eframe::run_native(
        "timber desk",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 480.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(desk))),
    )
}
