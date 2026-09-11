//! The timber desk: a little realtime mixing desk over the models.
//!
//! Architecture: drums and voice trigger like samples — the UI renders a
//! finished buffer with the same offline code the demo uses and ships it
//! to the audio thread. The strings are different: they are *streaming*
//! voices owned by the audio thread (timber::stream), persistent loops
//! the UI steers live through atomics — pluck them, bow them, change
//! their material while they ring. The mixer (gain, pan, mute, master
//! with a tanh limiter) is always live.
//!
//! Keys: Z X C V B N = drums · A S D F G H J K = pluck strings ·
//! Q W E R T = vowels. Bow the strings by dragging on the bow surface:
//! horizontal speed is bow speed, vertical position is pressure.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

use timber::modal::{self, Hit};
use timber::stream;
use timber::util::{AtomicF32, Rng, SR};
use timber::voice::{self, Note, Vowel};

// ---- Lock-free mixer state shared with the audio thread -----------------

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

/// Points in the string-motion scope the audio thread keeps refreshed.
const SCOPE_LEN: usize = 128;

enum Msg {
    /// A finished render for the buffer players (drums, voice).
    Buffer(usize, Vec<f32>, Option<u8>),
    /// Pluck one streaming string.
    Pluck(usize),
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
    scope: Arc<Mutex<Vec<f32>>>,
    rx: Receiver<Msg>,
    string_freqs: Vec<f32>,
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
    let mut rng = Rng(0x626f7765);
    let mut voices: Vec<PlayVoice> = Vec::new();
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let p = stream::Params::read(&ctl);
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        Msg::Pluck(i) => bank.pluck(i, ctl.pluck_pos.get(), &mut rng),
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
                    let (mut l, mut r) = (0.0f32, 0.0f32);
                    let mut route = |s: f32, strip: &Strip| {
                        if !strip.mute.load(Relaxed) {
                            // Equal-power pan.
                            let a = (strip.pan.get() + 1.0) * std::f32::consts::FRAC_PI_4;
                            let g = strip.gain.get() * s;
                            l += g * a.cos();
                            r += g * a.sin();
                        }
                    };
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
                        route(s, &mixer.strips[v.strip]);
                        true
                    });
                    route(bank.tick(&p), &mixer.strips[STRING]);
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
    damp: f32,
    noise: f32,
    noise_decay: f32,
    drive: f32,
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
        damp: 1.0,
        noise: 0.0,
        noise_decay: 0.1,
        drive: 0.0,
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
            duration: 0.5,
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
    ctl: Arc<stream::Ctl>,
    scope: Arc<Mutex<Vec<f32>>>,
    tx: Sender<Msg>,
    _stream: cpal::Stream,
    rng: Rng,

    pads: Vec<DrumParams>,
    pad_sel: usize,

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
            damp: p.damp,
            noise: p.noise,
            noise_decay: p.noise_decay.max(0.002),
            drive: p.drive,
            level: p.level,
        };
        let _ = self.tx.send(Msg::Buffer(
            DRUMS,
            modal::render(&hit, &mut self.rng),
            p.choke,
        ));
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
) {
    ui.spacing_mut().slider_width = track_width(ui);
    ui.add(egui::Slider::new(v, range).logarithmic(log).text(label));
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
        // Real key presses only: OS key-repeat would machine-gun plucks
        // and drum hits when a key is held.
        let mut acts: Vec<(usize, usize)> = Vec::new(); // (kind, index)
        ctx.input(|i| {
            for ev in &i.events {
                let egui::Event::Key {
                    key,
                    pressed: true,
                    repeat: false,
                    ..
                } = ev
                else {
                    continue;
                };
                if let Some(k) = DRUM_KEYS.iter().position(|d| d == key) {
                    acts.push((0, k));
                } else if let Some(k) = NOTE_KEYS.iter().position(|d| d == key) {
                    acts.push((1, k));
                } else if let Some(k) = VOWEL_KEYS.iter().position(|d| d == key) {
                    acts.push((2, k));
                }
            }
        });
        for (kind, k) in acts {
            match kind {
                0 => self.strike(k),
                // Left hand fingers, right hand excites: while the bow
                // is on the string, a key only changes the fingered
                // note under the sustained stroke — no pluck. With the
                // bow lifted, keys pluck as before.
                1 if self.ctl.bow_on.load(Relaxed) => {
                    self.ctl.bow_string.store(k, Relaxed);
                }
                1 => self.pluck(k),
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
            ui.small("A S D F G H J K — pluck (finger, while bowing)");
            ui.small("Q W E R T — vowels");
            ui.small("drag bow surface — bow");
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
                    .width((ui.available_width() - 130.0).clamp(80.0, 160.0))
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
                slider(ui, &mut p.damp, 0.1..=3.0, true, "muffle");
                slider(ui, &mut p.noise, 0.0..=1.0, false, "rattle");
                slider(ui, &mut p.noise_decay, 0.002..=0.5, true, "rattle decay");
                slider(ui, &mut p.drive, 0.0..=6.0, false, "drive");
                slider(ui, &mut p.level, 0.0..=1.0, false, "level");
                let mut chokes = p.choke.is_some();
                if ui.checkbox(&mut chokes, "choke group").changed() {
                    p.choke = if chokes { Some(0) } else { None };
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
                ui.label(format!(
                    "bow — {} (hold: ⇄ from center = speed, height = pressure)",
                    NOTES[bowed].0
                ));
                let h = 80.0;
                let (rect, resp) = ui
                    .allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::drag());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
                if let Ok(shape) = self.scope.lock() {
                    let pts: Vec<egui::Pos2> = shape
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            egui::pos2(
                                rect.left() + rect.width() * i as f32 / (shape.len() - 1) as f32,
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
    let ctl = Arc::new(stream::Ctl::new());
    let scope = Arc::new(Mutex::new(vec![0.0f32; SCOPE_LEN]));
    let (tx, rx) = channel();
    let freqs = NOTES.iter().map(|(_, f)| *f).collect();
    let stream = start_audio(mixer.clone(), ctl.clone(), scope.clone(), rx, freqs);

    let desk = Desk {
        mixer,
        ctl,
        scope,
        tx,
        _stream: stream,
        rng: Rng(0x74696d62),
        pads: default_pads(),
        pad_sel: 0,
        note: Note::default(),
        from_vowel: 1, // ee → ah: "yah"
        to_vowel: 0,
        voice_freq: 165.0,
    };

    eframe::run_native(
        "timber desk",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1020.0, 520.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(desk))),
    )
}
