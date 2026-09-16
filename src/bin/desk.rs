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
//! Keys: the number keys pick a bank (1 modal kit, 2 physical kit,
//! 3 percussion, 4–7 tuned instruments on the melody keys; tab flips
//! modal ↔ physical) and the bottom row Z..= / plays it — the two kits
//! mirror each other: kick snare toms | hat open-hat | crash ride |
//! clap or foot (shift = roll on the modal kit, hard hit on the
//! physical) · space = loop ·
//! A S D F G H J K =
//! pluck strings (finger while bowing; melody while a voice engine is
//! held or speaking) · Q W E R T = vowels (steer the held engine).
//! Bow surface: hold, x-offset from center = speed, height = pressure.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

use timber::looper::{self, Looper};
use timber::util::{AtomicF32, Rng, SR};
use timber::voice::{self, Note, Vowel};
use timber::{body, cymbal, drums, hihat, mesh, modal, mouth, sing, speak, stream, tract};

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

/// What the views draw, refreshed by the audio thread ~30×/s: the
/// selected mesh head and cymbal plate as displacement fields, and the
/// selected modal pad's modes as (Hz, table gain, live amplitude).
struct ViewData {
    mesh: Vec<f32>,
    cymbal: Vec<f32>,
    /// Mean-square displacement per cell since the last refresh: the
    /// energy view, which does not flicker (an instantaneous surface
    /// oscillating at hundreds of Hz, sampled 30×/s, is a random
    /// phase each frame) and shows the mode shapes as Chladni did —
    /// bright lobes, dark nodal lines.
    mesh_energy: Vec<f32>,
    cymbal_energy: Vec<f32>,
    ladder: Vec<(f32, f32, f32)>,
    /// The pad's noise burst: (level, tone, wash center Hz).
    noise: (f32, f32, f32),
}

/// Which pads the views should watch (the UI's current selections).
struct ViewCtl {
    pad: AtomicUsize,
    mesh: AtomicUsize,
    cymbal: AtomicUsize,
    /// Accumulate energy (else only the instantaneous surface).
    energy: AtomicBool,
}

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
    /// The looper's one button, undo and clear.
    Loop(LoopCmd),
    /// A modal pad at a pitch (the melody keys), its roll on/off, and
    /// its key coming up (the damper, if the pad has one).
    Note(usize, f32),
    NoteRoll(usize, f32, bool),
    NoteOff(usize, f32),
    /// Hi-hat: stick on the top plate, the pedal, its params.
    HatStrike(f32),
    HatPedal(bool),
    HatPad(hihat::HiHatParams),
}

#[derive(Clone, Copy)]
enum LoopCmd {
    Toggle,
    Stop,
    Undo,
    Clear,
}

/// What the looper records: a hit on a pad, or a roll's start/stop.
/// Knob moves are deliberately not events — they are the live layer
/// over the loop.
#[derive(Clone, Copy)]
enum Hit {
    Pad(usize),
    Mesh(usize, f32),
    Cymbal(usize, f32),
    Pluck(usize),
    Roll(usize, bool),
    Hat(f32),
    Pedal(bool),
    Note(usize, f32),
    NoteRoll(usize, f32, bool),
    NoteOff(usize, f32),
}

/// Looper settings and its state for display, shared with the audio
/// thread (which owns the loop itself, on its sample clock).
struct LoopCtl {
    bpm: AtomicF32,
    quantize: AtomicBool,
    click: AtomicBool,
    /// Tap-tempo armed for the next recording.
    tap: AtomicBool,
    /// looper::State as usize, layers, playhead fraction, length s.
    state: AtomicUsize,
    layers: AtomicUsize,
    pos: AtomicF32,
    len: AtomicF32,
}

/// Everything a hit can land on.
struct Instruments {
    bank: stream::Bank,
    kit: drums::Kit,
    heads: mesh::Kit,
    cymbals: cymbal::Kit,
    hat: hihat::HiHat,
}

impl Instruments {
    fn dispatch(&mut self, hit: Hit, pluck_pos: f32, rng: &mut Rng) {
        match hit {
            Hit::Pad(i) => self.kit.strike(i),
            Hit::Mesh(i, s) => self.heads.strike(i, s),
            Hit::Cymbal(i, s) => self.cymbals.strike(i, s),
            Hit::Pluck(i) => self.bank.pluck(i, pluck_pos, rng),
            Hit::Roll(i, on) => self.kit.set_roll(i, on),
            Hit::Hat(s) => self.hat.strike(s),
            Hit::Pedal(down) => self.hat.pedal(down),
            Hit::Note(i, f) => self.kit.strike_note(i, f),
            Hit::NoteRoll(i, f, on) => self.kit.set_note_roll(i, f, on),
            Hit::NoteOff(i, f) => self.kit.release_note(i, f),
        }
    }
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

#[allow(clippy::too_many_arguments)] // one call site; every argument is a distinct shared handle
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
    loop_ctl: Arc<LoopCtl>,
    view: Arc<Mutex<ViewData>>,
    view_ctl: Arc<ViewCtl>,
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

    let mut ins = Instruments {
        bank: stream::Bank::new(&string_freqs),
        kit: drums::Kit::new(pads),
        heads: mesh::Kit::new(mesh_pads),
        cymbals: cymbal::Kit::new(modes.clone(), cymbal_pads),
        hat: hihat::HiHat::new(modes, hihat::default_params()),
    };
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
    // The looper lives here, on the output frame clock, so its
    // timing is sample-exact whatever the UI thread's jitter.
    let mut looper: Looper<Hit> = Looper::new();
    // Energy accumulators for the surface views, summed every few
    // frames and handed over at the next refresh.
    let mut acc_mesh = vec![0.0f32; mesh::Mesh::width() * mesh::Mesh::width()];
    let mut acc_cymbal = vec![0.0f32; cymbal::Cymbal::width() * cymbal::Cymbal::width()];
    let mut acc_scratch = vec![0.0f32; cymbal::Cymbal::width() * cymbal::Cymbal::width()];
    let mut acc_n = 0u32;
    let mut clock: u64 = 0;
    let mut due: Vec<Hit> = Vec::new();
    // Metronome: a short sine ping, higher on beat one.
    let (mut click_env, mut click_ph, mut click_f) = (0.0f32, 0.0f32, 1400.0f32);
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let p = stream::Params::read(&ctl);
                let mp = mouth::Params::read(&mouth_ctl);
                let tp = tract::Params::read(&tract_ctl);
                looper.bpm = loop_ctl.bpm.get();
                looper.quantize = loop_ctl.quantize.load(Relaxed);
                looper.tap = loop_ctl.tap.load(Relaxed);
                while let Ok(msg) = rx.try_recv() {
                    // Hits go through the looper (which records them if
                    // it is listening) and then to the instruments.
                    let hit = match msg {
                        Msg::Pluck(i) => Some(Hit::Pluck(i)),
                        Msg::Strike(i) => Some(Hit::Pad(i)),
                        Msg::Roll(i, on) => Some(Hit::Roll(i, on)),
                        Msg::MeshStrike(i, s) => Some(Hit::Mesh(i, s)),
                        Msg::CymbalStrike(i, s) => Some(Hit::Cymbal(i, s)),
                        Msg::HatStrike(s) => Some(Hit::Hat(s)),
                        Msg::HatPedal(down) => Some(Hit::Pedal(down)),
                        Msg::Note(i, f) => Some(Hit::Note(i, f)),
                        Msg::NoteRoll(i, f, on) => Some(Hit::NoteRoll(i, f, on)),
                        Msg::NoteOff(i, f) => Some(Hit::NoteOff(i, f)),
                        _ => None,
                    };
                    if let Some(h) = hit {
                        looper.record(clock, h);
                        ins.dispatch(h, ctl.pluck_pos.get(), &mut rng);
                        continue;
                    }
                    match msg {
                        Msg::Pluck(_)
                        | Msg::Strike(_)
                        | Msg::Roll(..)
                        | Msg::MeshStrike(..)
                        | Msg::CymbalStrike(..)
                        | Msg::HatStrike(_)
                        | Msg::HatPedal(_)
                        | Msg::Note(..)
                        | Msg::NoteRoll(..)
                        | Msg::NoteOff(..) => unreachable!(),
                        Msg::HatPad(params) => ins.hat.set_params(params),
                        Msg::Loop(LoopCmd::Toggle) => {
                            if looper.toggle(clock) {
                                // A tapped tempo: hand it back to the UI.
                                loop_ctl.bpm.set(looper.bpm);
                                loop_ctl.quantize.store(true, Relaxed);
                                loop_ctl.tap.store(false, Relaxed);
                            }
                        }
                        Msg::Loop(LoopCmd::Stop) => looper.stop(clock),
                        Msg::Loop(LoopCmd::Undo) => looper.undo(),
                        Msg::Loop(LoopCmd::Clear) => looper.clear(),
                        Msg::Pad(i, params) => ins.kit.set_params(i, params),
                        Msg::Speak(segs) => utter = speak::Utterance::new(segs),
                        Msg::MeshPad(i, params) => ins.heads.set_params(i, params),
                        Msg::CymbalPad(i, params) => ins.cymbals.set_params(i, params),
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
                    // The loop replays its hits on this clock.
                    due.clear();
                    looper.due(clock, &mut due);
                    for h in due.drain(..) {
                        ins.dispatch(h, ctl.pluck_pos.get(), &mut rng);
                    }
                    if loop_ctl.click.load(Relaxed)
                        && let Some(one) = looper.beat(clock)
                    {
                        click_env = 1.0;
                        click_ph = 0.0;
                        click_f = if one { 2000.0 } else { 1400.0 };
                    }
                    if clock.is_multiple_of(256) {
                        if let Some(bpm) = looper.tap_bpm()
                            && looper.state() == looper::State::Recording
                        {
                            loop_ctl.bpm.set(bpm);
                        }
                        loop_ctl.state.store(looper.state() as usize, Relaxed);
                        loop_ctl.layers.store(looper.layers(), Relaxed);
                        loop_ctl.pos.set(looper.position(clock));
                        loop_ctl.len.set(looper.len_secs());
                    }
                    if view_ctl.energy.load(Relaxed) && clock.is_multiple_of(8) {
                        if let Some(u) = ins.heads.surface(view_ctl.mesh.load(Relaxed)) {
                            for (a, v) in acc_mesh.iter_mut().zip(u) {
                                *a += v * v;
                            }
                        }
                        ins.cymbals
                            .surface(view_ctl.cymbal.load(Relaxed), &mut acc_scratch);
                        for (a, v) in acc_cymbal.iter_mut().zip(&acc_scratch) {
                            *a += v * v;
                        }
                        acc_n += 1;
                    }
                    clock += 1;
                    // Sum each strip dry first: the body must see the
                    // strip's whole signal once, not each voice.
                    let mut sums = [0.0f32; 3];
                    if click_env > 1e-4 {
                        sums[DRUMS] += click_ph.sin() * click_env * 0.25;
                        click_ph += std::f32::consts::TAU * click_f / SR;
                        click_env *= (-1.0 / (0.006 * SR)).exp();
                    }
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
                    sums[STRING] += ins.bank.tick(&p);
                    sums[DRUMS] += ins.kit.tick(&mut rng)
                        + ins.heads.tick(&mut rng)
                        + ins.cymbals.tick(&mut rng)
                        + ins.hat.tick(&mut rng);
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
                    ins.bank.shape(p.bow_string, &mut s);
                }
                if let Ok(mut v) = view.try_lock() {
                    if let Some(u) = ins.heads.surface(view_ctl.mesh.load(Relaxed)) {
                        v.mesh.copy_from_slice(u);
                    }
                    ins.cymbals
                        .surface(view_ctl.cymbal.load(Relaxed), &mut v.cymbal);
                    if acc_n > 0 {
                        let k = 1.0 / acc_n as f32;
                        for (e, a) in v.mesh_energy.iter_mut().zip(acc_mesh.iter_mut()) {
                            *e = *a * k;
                            *a = 0.0;
                        }
                        for (e, a) in v.cymbal_energy.iter_mut().zip(acc_cymbal.iter_mut()) {
                            *e = *a * k;
                            *a = 0.0;
                        }
                        acc_n = 0;
                    }
                    let sel = view_ctl.pad.load(Relaxed);
                    let mut ladder = std::mem::take(&mut v.ladder);
                    ins.kit.mode_levels(sel, &mut ladder);
                    v.ladder = ladder;
                    v.noise = ins.kit.noise_view(sel);
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
    Gong,
    Hat,
    Snare,
    Marimba,
    Vibes,
    Pan,
    Cowbell,
    Woodblock,
    Tambourine,
    Simmons,
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
            ModeSet::Gong => modal::GONG,
            ModeSet::Hat => modal::HAT,
            ModeSet::Snare => modal::SNARE,
            ModeSet::Marimba => modal::MARIMBA,
            ModeSet::Vibes => modal::VIBES,
            ModeSet::Pan => modal::PAN,
            ModeSet::Cowbell => modal::COWBELL,
            ModeSet::Woodblock => modal::WOODBLOCK,
            ModeSet::Tambourine => modal::TAMBOURINE,
            ModeSet::Simmons => modal::SIMMONS,
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
            ModeSet::Gong => "gong (synth)",
            ModeSet::Hat => "hi-hat (synth)",
            ModeSet::Snare => "snare head",
            ModeSet::Marimba => "marimba bar",
            ModeSet::Vibes => "vibraphone bar",
            ModeSet::Pan => "steel pan",
            ModeSet::Cowbell => "cowbell",
            ModeSet::Woodblock => "wood block",
            ModeSet::Tambourine => "tambourine",
            ModeSet::Simmons => "Simmons tom",
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
    bloom_delay: f32,
    bloom_spread: f32,
    drive: f32,
    shimmer: f32,
    attack: f32,
    claps: u8,
    tremolo: f32,
    bloom_lo: f32,
    bleed: f32,
    roll_rate: f32,
    roll_strength: f32,
    damper: bool,
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
            bloom_delay: self.bloom_delay,
            bloom_spread: self.bloom_spread,
            drive: self.drive,
            shimmer: self.shimmer,
            attack: self.attack,
            claps: self.claps,
            tremolo: self.tremolo,
            bloom_lo: self.bloom_lo,
            bleed: self.bleed,
            roll_rate: self.roll_rate,
            roll_strength: self.roll_strength,
            damper: self.damper,
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
        bloom_delay: 0.025,
        bloom_spread: 0.0,
        drive: 0.0,
        shimmer: 0.0,
        attack: 0.0,
        claps: 1,
        tremolo: 0.0,
        bloom_lo: 0.0,
        bleed: 0.0,
        roll_rate: 14.0,
        roll_strength: 0.75,
        damper: false,
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
            modes: ModeSet::Snare,
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
        // Synth hats: the HAT partials under a bright noise burst, the
        // closed one muffled hard (the pressed pair), the open one
        // ringing with a wash and a little shimmer. Same choke group,
        // so the closed hat cuts the open one — the pedal, by rule.
        DrumParams {
            name: "hat",
            modes: ModeSet::Hat,
            freq: 600.0,
            damp: 0.08,
            noise: 0.9,
            noise_decay: 0.03,
            noise_tone: 0.3,
            level: 0.35,
            choke: Some(0),
            ..base
        },
        DrumParams {
            name: "open hat",
            modes: ModeSet::Hat,
            freq: 600.0,
            damp: 0.6,
            noise: 0.8,
            noise_decay: 0.25,
            noise_tone: 0.6,
            bloom: 0.3,
            shimmer: 0.8,
            level: 0.35,
            choke: Some(0),
            ..base
        },
        // Synth cymbals: the CYMBAL partial table with the pad's noise
        // burst as the wash. The crash is mostly wash, the ride mostly
        // partials with a short sizzle.
        DrumParams {
            name: "crash",
            modes: ModeSet::Cymbal,
            freq: 450.0,
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
        // The 808's handclap: four bursts of bandpassed noise, the last
        // with a tail.
        DrumParams {
            name: "clap",
            modes: ModeSet::NoiseOnly,
            noise: 1.0,
            noise_decay: 0.1,
            noise_tone: 0.7,
            claps: 4,
            level: 0.5,
            ..base
        },
        // A cowbell at the 808's pitch, struck with a stick: a hard,
        // bright knock and a short clang.
        DrumParams {
            name: "cowbell",
            modes: ModeSet::Cowbell,
            freq: 540.0,
            noise: 0.25,
            noise_decay: 0.003,
            drive: 0.8,
            level: 0.5,
            ..base
        },
        // A wood block: the knock is most of it.
        DrumParams {
            name: "wood block",
            modes: ModeSet::Woodblock,
            freq: 800.0,
            noise: 0.5,
            noise_decay: 0.004,
            noise_tone: 0.3,
            level: 0.55,
            ..base
        },
        // Tambourine: the jingle cluster under a bright rattle burst.
        DrumParams {
            name: "tambourine",
            modes: ModeSet::Tambourine,
            // From recordings (samples/tamb_*): a hit is *short* — −17 dB
            // by 100 ms, −39 by 400 — and stays bright, because what
            // lingers is the jingles' ping at 3–6 kHz, not noise. So:
            // a short flat burst, clashed twice (the jingles bounce
            // once ~5 ms after the hit), over an unmuffled cluster.
            freq: 3800.0,
            noise: 1.0,
            noise_decay: 0.035,
            noise_tone: 0.1,
            claps: 2,
            shimmer: 0.5,
            level: 0.45,
            ..base
        },
        // Shaker: nothing but the rattle — a short burst of the
        // brightest wash, no partials at all.
        DrumParams {
            name: "shaker",
            modes: ModeSet::NoiseOnly,
            noise: 1.0,
            noise_decay: 0.045,
            noise_tone: 0.9,
            level: 0.35,
            ..base
        },
        // The Simmons disco tom: a big downward sweep on a single
        // resonance, a click on the front, some drive. Play it from
        // the melody keys for the descending fill (bank 8).
        DrumParams {
            name: "disco tom",
            modes: ModeSet::Simmons,
            freq: 150.0,
            glide: 0.7,
            glide_time: 0.16,
            noise: 0.35,
            noise_decay: 0.003,
            drive: 1.2,
            level: 0.7,
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
        // Tam-tam: low, loud lows, ten-second decays, and the swell —
        // the bloom arriving 150 ms after the strike over 400 ms.
        DrumParams {
            name: "gong",
            modes: ModeSet::Gong,
            freq: 55.0,
            noise: 0.15,
            noise_decay: 0.3,
            noise_tone: 1.0,
            bloom: 0.9,
            bloom_delay: 0.15,
            bloom_spread: 0.4,
            drive: 0.4,
            shimmer: 0.6,
            level: 0.5,
            ..base
        },
        // Tuned: `freq` is the pitch of the A key with the melody keys
        // on (middle C by default); the object carries the tuning.
        DrumParams {
            name: "marimba",
            modes: ModeSet::Marimba,
            freq: 261.63,
            attack: 0.003,
            // The mallet's knock on the wood: quiet, dull, and a
            // little longer than a click — yarn on rosewood, not a
            // stick on a block.
            noise: 0.15,
            noise_decay: 0.005,
            noise_tone: 0.6,
            drive: 0.3,
            // Mallet rolls are softer than stick rolls.
            roll_strength: 0.5,
            level: 0.9,
            ..base
        },
        DrumParams {
            name: "vibes",
            modes: ModeSet::Vibes,
            freq: 261.63,
            attack: 0.003,
            tremolo: 0.5,
            bleed: 0.08,
            roll_strength: 0.5,
            damper: true,
            level: 0.8,
            ..base
        },
        // The pan: a hard note starts sharp and settles (the dome
        // stiffening), the octave and twelfth bloom in just after, the
        // neighbours ring along, and the pairs beat.
        DrumParams {
            name: "steel pan",
            modes: ModeSet::Pan,
            freq: 261.63,
            glide: 0.02,
            glide_time: 0.06,
            // A rubber-tipped stick: near-instant contact (a 1.5 ms
            // mallet push has its spectral null at 1 kHz — right on the
            // pan's loudest partial), so the base's instant attack.
            // The stick on the steel: a 2 kHz-centroid impact the
            // recordings show in the first 5 ms.
            noise: 1.0,
            noise_decay: 0.004,
            bloom: 0.5,
            bloom_delay: 0.012,
            bloom_spread: 0.03,
            bloom_lo: 1.0,
            bleed: 0.08,
            shimmer: 0.25,
            // Two sticks: soft, even, quick.
            roll_rate: 15.0,
            roll_strength: 0.45,
            level: 0.7,
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

/// The bottom row plays whichever bank is selected; the number keys
/// select the bank. Keys are matched against the *physical* key when
/// the event carries one (shift can change the logical key on some
/// layouts — QWERTZ turns shift+',' into ';' — which would strand roll
/// state).
const ROW_KEYS: [egui::Key; 10] = [
    egui::Key::Z,
    egui::Key::X,
    egui::Key::C,
    egui::Key::V,
    egui::Key::B,
    egui::Key::N,
    egui::Key::M,
    egui::Key::Comma,
    egui::Key::Period,
    egui::Key::Slash,
];
const NROW: usize = ROW_KEYS.len();
const BANK_KEYS: [egui::Key; 8] = [
    egui::Key::Num1,
    egui::Key::Num2,
    egui::Key::Num3,
    egui::Key::Num4,
    egui::Key::Num5,
    egui::Key::Num6,
    egui::Key::Num7,
    egui::Key::Num8,
];

/// A bank: what the bottom row plays. The two kits mirror each other
/// slot for slot (kick snare toms | hat open-hat | crash ride | extra)
/// so a beat transfers between them with one key — and the hats sit
/// under the first fingers of the right hand, where the eighths go.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bank {
    /// The modal kit (shift = roll).
    Modal,
    /// The physical kit: mesh drums, modal-plate cymbals, the two-plate
    /// hi-hat with its foot on the last key (shift = hard hit).
    Physical,
    /// The modal kit's other percussion: bell, triangle, gong, clap,
    /// cowbell, wood block, tambourine, shaker, Simmons disco tom.
    Percussion,
    /// A tuned modal pad on the melody keys.
    Tuned(&'static str),
}

const BANKS: [Bank; 8] = [
    Bank::Modal,
    Bank::Physical,
    Bank::Percussion,
    Bank::Tuned("marimba"),
    Bank::Tuned("vibes"),
    Bank::Tuned("steel pan"),
    Bank::Tuned("bell"),
    Bank::Tuned("disco tom"),
];

impl Bank {
    fn name(self) -> String {
        match self {
            Bank::Modal => "modal kit".into(),
            Bank::Physical => "physical kit".into(),
            Bank::Percussion => "percussion".into(),
            Bank::Tuned(n) => n.into(),
        }
    }
}

/// What one bottom-row key does in the current bank.
#[derive(Clone, Copy, PartialEq)]
enum Slot {
    None,
    /// A modal pad, by index into the pad list.
    Pad(usize),
    Mesh(usize),
    Cymbal(usize),
    HatClosed,
    HatOpen,
    Foot,
}

const MODAL_ROW: [&str; 10] = [
    "kick", "snare", "tom hi", "tom", "tom lo", "hat", "open hat", "crash", "ride", "clap",
];
/// The modal objects that don't fit on the kit row; the rest of the
/// row is empty rather than repeating bank 1.
const PERC_ROW: [&str; 10] = [
    "bell",
    "triangle",
    "gong",
    "clap",
    "cowbell",
    "wood block",
    "tambourine",
    "shaker",
    "disco tom",
    "",
];
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
    hat: hihat::HiHatParams,
    pedal_down: bool,
    /// The home row plays the selected modal pad chromatically.
    melody: bool,
    octave: i32,
    /// Melody keys held with shift: (semitone, pad, freq) rolling.
    note_rolls: Vec<(usize, usize, f32)>,
    /// Melody keys held: (semitone, pad, freq), for the damper on
    /// release.
    held_notes: Vec<(usize, usize, f32)>,
    /// The pedal: no damper on release while on.
    sustain: bool,
    loop_ctl: Arc<LoopCtl>,
    view: Arc<Mutex<ViewData>>,
    view_ctl: Arc<ViewCtl>,
    /// Running peaks the views normalize against (decaying, so a
    /// quiet tail still shows).
    gain_mesh: f32,
    gain_cymbal: f32,
    gain_ladder: f32,
    bank: Bank,
    /// The bottom row: which keys are held, and which pad each held
    /// key is rolling (shift), by row slot.
    row_down: [bool; NROW],
    row_rolling: [Option<usize>; NROW],
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

    fn pad_named(&self, name: &str) -> Option<usize> {
        self.pads.iter().position(|p| p.name == name)
    }

    /// The bottom row's ten slots for the current bank.
    fn slots(&self) -> [Slot; NROW] {
        let mut s = [Slot::None; NROW];
        match self.bank {
            Bank::Modal | Bank::Percussion => {
                let names = if self.bank == Bank::Modal {
                    MODAL_ROW
                } else {
                    PERC_ROW
                };
                for (k, n) in names.iter().enumerate() {
                    if !n.is_empty()
                        && let Some(i) = self.pad_named(n)
                    {
                        s[k] = Slot::Pad(i);
                    }
                }
            }
            Bank::Physical => {
                for (k, slot) in s.iter_mut().enumerate().take(5) {
                    *slot = Slot::Mesh(k);
                }
                let cym = |name: &str| {
                    self.cymbal_pads
                        .iter()
                        .position(|(n, _)| *n == name)
                        .map(Slot::Cymbal)
                        .unwrap_or(Slot::None)
                };
                s[5] = Slot::HatClosed;
                s[6] = Slot::HatOpen;
                s[7] = cym("crash");
                s[8] = cym("ride");
                s[9] = Slot::Foot;
            }
            Bank::Tuned(_) => {}
        }
        s
    }

    /// Switch bank: stop anything the row was holding, and put the
    /// melody keys on a tuned bank's pad.
    fn select_bank(&mut self, bank: Bank) {
        self.release_row();
        self.bank = bank;
        match bank {
            Bank::Tuned(name) => {
                if let Some(i) = self.pad_named(name) {
                    self.pad_sel = i;
                }
                self.melody = true;
            }
            _ => self.melody = false,
        }
    }

    /// Everything the bottom row is holding, let go (bank switch, a
    /// text field taking the keyboard).
    fn release_row(&mut self) {
        self.row_down = [false; NROW];
        for k in 0..NROW {
            if let Some(pad) = self.row_rolling[k].take() {
                let _ = self.tx.send(Msg::Roll(pad, false));
            }
        }
        if self.pedal_down {
            self.pedal_down = false;
            let _ = self.tx.send(Msg::HatPedal(false));
        }
    }

    fn strike_mesh(&mut self, pad: usize, hard: bool) {
        self.mesh_sel = pad;
        let strength = if hard { 1.6 } else { 0.8 };
        let _ = self.tx.send(Msg::MeshStrike(pad, strength));
    }

    fn strike_cymbal(&mut self, pad: usize, hard: bool) {
        self.cymbal_sel = pad;
        // A gentler "hard" than the drums': a cymbal's ring sums over
        // hits, and 1.6 repeated ran the limiter into static.
        let strength = if hard { 1.25 } else { 0.8 };
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

/// A key as it is printed on the cap.
fn key_label(key: egui::Key) -> &'static str {
    use egui::Key::*;
    match key {
        Z => "Z",
        X => "X",
        C => "C",
        V => "V",
        B => "B",
        N => "N",
        M => "M",
        Comma => ",",
        Period => ".",
        Slash => "/",
        _ => "?",
    }
}

/// The chromatic keyboard for the melody keys: white keys along the
/// home row from A (C), black keys on the row above.
fn melody_semitone(key: egui::Key) -> Option<usize> {
    use egui::Key::*;
    Some(match key {
        A => 0,
        W => 1,
        S => 2,
        E => 3,
        D => 4,
        F => 5,
        T => 6,
        G => 7,
        Y => 8,
        H => 9,
        U => 10,
        J => 11,
        K => 12,
        O => 13,
        L => 14,
        P => 15,
        Semicolon => 16,
        Quote => 17,
        _ => return None,
    })
}

/// A field on a square, normalized by a decaying running peak. Motion:
/// blue down, red up, on the panel ground. Energy: mean-square
/// displacement, dark to hot — the Chladni figure, with nodal lines
/// dark, and no flicker.
fn draw_surface(
    ui: &mut egui::Ui,
    field: &[f32],
    w: usize,
    size: f32,
    gain: &mut f32,
    energy: bool,
) {
    let peak = field.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    *gain = (*gain * 0.97).max(peak).max(1e-9);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let bg = ui.visuals().extreme_bg_color;
    painter.rect_filled(rect, 4.0, bg);
    let cell = size / w as f32;
    let up = egui::Color32::from_rgb(255, 120, 60);
    let down = egui::Color32::from_rgb(70, 140, 255);
    let hot = egui::Color32::from_rgb(255, 230, 120);
    let warm = egui::Color32::from_rgb(200, 60, 40);
    for j in 0..w {
        for i in 0..w {
            let v = (field[j * w + i] / *gain).clamp(-1.0, 1.0);
            if v.abs() < 0.02 {
                continue;
            }
            let c = if energy {
                // Square root so the quiet lobes read too.
                let t = v.max(0.0).sqrt();
                if t < 0.5 {
                    bg.lerp_to_gamma(warm, t * 2.0)
                } else {
                    warm.lerp_to_gamma(hot, (t - 0.5) * 2.0)
                }
            } else if v > 0.0 {
                bg.lerp_to_gamma(up, v)
            } else {
                bg.lerp_to_gamma(down, -v)
            };
            let p = egui::pos2(rect.left() + i as f32 * cell, rect.top() + j as f32 * cell);
            painter.rect_filled(
                egui::Rect::from_min_size(p, egui::vec2(cell + 0.5, cell + 0.5)),
                0.0,
                c,
            );
        }
    }
}

/// The mode ladder: each mode a bar on a log-frequency axis, grey to
/// its table gain, lit to its live amplitude (normalized by a decaying
/// running peak). Shimmer partners draw beside their primaries.
fn draw_ladder(
    ui: &mut egui::Ui,
    modes: &[(f32, f32, f32)],
    noise: (f32, f32, f32),
    gain: &mut f32,
) {
    let h = 70.0;
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let (lo, hi) = (40.0f32.ln(), 16000.0f32.ln());
    let x_of = |f: f32| rect.left() + rect.width() * ((f.max(40.0).ln() - lo) / (hi - lo));
    // Ticks at 100 Hz, 1 kHz, 10 kHz.
    for f in [100.0, 1000.0, 10000.0] {
        painter.vline(
            x_of(f),
            rect.y_range(),
            egui::Stroke::new(1.0, ui.visuals().weak_text_color().gamma_multiply(0.4)),
        );
    }
    // The noise burst, so the picture is honest about where a pad's
    // sound comes from: a translucent band, rising to the right for
    // the flat (differenced) rattle, a hump around the wash's sliding
    // resonance for the toned one, its height the burst's level.
    let (nl, tone, fc) = noise;
    if nl > 0.01 {
        let strips = 48;
        let band = egui::Color32::from_rgb(120, 200, 160);
        for k in 0..strips {
            let t0 = k as f32 / strips as f32;
            let t1 = (k + 1) as f32 / strips as f32;
            let f = (lo + (hi - lo) * (t0 + t1) * 0.5).exp();
            let flat = 0.15 + 0.85 * t0;
            let oct = (f / fc).ln() / std::f32::consts::LN_2;
            let hump = (-0.5 * oct * oct).exp();
            let a = (nl * ((1.0 - tone) * flat + tone * hump)).clamp(0.0, 1.0);
            let x0 = rect.left() + rect.width() * t0;
            let x1 = rect.left() + rect.width() * t1;
            let top = rect.bottom() - 2.0 - (h - 6.0) * a;
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, top), egui::pos2(x1, rect.bottom() - 2.0)),
                0.0,
                band.gamma_multiply(0.35),
            );
        }
    }
    let peak = modes.iter().fold(0.0f32, |m, v| m.max(v.2));
    *gain = (*gain * 0.95).max(peak).max(1e-5);
    let gmax = modes.iter().fold(0.0f32, |m, v| m.max(v.1)).max(1e-6);
    for &(f, g, a) in modes {
        if f > 16000.0 {
            continue;
        }
        let x = x_of(f);
        let y0 = rect.bottom() - 2.0;
        let gh = (h - 6.0) * (g / gmax);
        painter.line_segment(
            [egui::pos2(x, y0), egui::pos2(x, y0 - gh)],
            egui::Stroke::new(2.0, ui.visuals().weak_text_color().gamma_multiply(0.5)),
        );
        let ah = (h - 6.0) * (a / *gain).clamp(0.0, 1.0);
        if ah > 0.5 {
            painter.line_segment(
                [egui::pos2(x, y0), egui::pos2(x, y0 - ah)],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 190, 80)),
            );
        }
    }
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
                self.release_row();
                for (_, pad, f) in self.note_rolls.drain(..) {
                    let _ = self.tx.send(Msg::NoteRoll(pad, f, false));
                }
                for (_, pad, f) in self.held_notes.drain(..) {
                    let _ = self.tx.send(Msg::NoteOff(pad, f));
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
                if *pressed && !*repeat && key == egui::Key::Tab {
                    acts.push((19, 0));
                } else if *pressed
                    && !*repeat
                    && let Some(b) = BANK_KEYS.iter().position(|d| *d == key)
                {
                    acts.push((20, b));
                } else if let Some(k) = ROW_KEYS.iter().position(|d| *d == key) {
                    let slot = self.slots()[k];
                    self.row_down[k] = *pressed;
                    match slot {
                        Slot::Foot => {
                            // The foot: down while held, up on release.
                            if !*repeat && *pressed != self.pedal_down {
                                self.pedal_down = *pressed;
                                acts.push((if *pressed { 12 } else { 13 }, 0));
                            }
                        }
                        Slot::Pad(p) if *pressed && !*repeat => acts.push((0, p)),
                        Slot::Mesh(p) if *pressed && !*repeat => {
                            acts.push((if i.modifiers.shift { 4 } else { 3 }, p))
                        }
                        Slot::Cymbal(p) if *pressed && !*repeat => {
                            acts.push((if i.modifiers.shift { 6 } else { 5 }, p))
                        }
                        Slot::HatClosed if *pressed && !*repeat => acts.push((11, 0)),
                        Slot::HatOpen if *pressed && !*repeat => acts.push((11, 1)),
                        _ => {}
                    }
                } else if *pressed && !*repeat && key == egui::Key::Space {
                    acts.push((if i.modifiers.shift { 10 } else { 7 }, 0));
                } else if *pressed && !*repeat && key == egui::Key::Backspace {
                    acts.push((if i.modifiers.shift { 9 } else { 8 }, 0));
                } else if self.melody
                    && !*repeat
                    && let Some(semi) = melody_semitone(key)
                {
                    if *pressed {
                        acts.push((14, semi));
                        if i.modifiers.shift {
                            acts.push((17, semi));
                        }
                    } else {
                        acts.push((18, semi));
                    }
                } else if self.melody && *pressed && !*repeat && key == egui::Key::OpenBracket {
                    acts.push((15, 0));
                } else if self.melody && *pressed && !*repeat && key == egui::Key::CloseBracket {
                    acts.push((16, 0));
                } else if *pressed && !*repeat {
                    if let Some(k) = NOTE_KEYS.iter().position(|d| *d == key) {
                        acts.push((1, k));
                    } else if let Some(k) = VOWEL_KEYS.iter().position(|d| *d == key) {
                        acts.push((2, k));
                    }
                }
            }
            // Shift + a held modal pad key = roll; a plain press stays a
            // clean single hit.
            let slots = self.slots();
            for (k, slot) in slots.iter().enumerate() {
                let want = match *slot {
                    Slot::Pad(p) if self.row_down[k] && i.modifiers.shift => Some(p),
                    _ => None,
                };
                if want != self.row_rolling[k] {
                    if let Some(pad) = self.row_rolling[k] {
                        let _ = self.tx.send(Msg::Roll(pad, false));
                    }
                    if let Some(pad) = want {
                        let _ = self.tx.send(Msg::Roll(pad, true));
                    }
                    self.row_rolling[k] = want;
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
                7 => {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Toggle));
                }
                8 => {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Undo));
                }
                9 => {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Clear));
                }
                10 => {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Stop));
                }
                // The hat as a pair of pads: `,` a closed hit, `.` an open
                // one — the key sets the foot, so a beat's closed eighths
                // and the odd open hit are one hand. (`/` held is the foot
                // itself: chick, and closing an open hat.)
                11 => {
                    let open = k == 1;
                    if !self.pedal_down {
                        let _ = self.tx.send(Msg::HatPedal(!open));
                    }
                    let _ = self.tx.send(Msg::HatStrike(0.9));
                }
                12 => {
                    let _ = self.tx.send(Msg::HatPedal(true));
                }
                13 => {
                    let _ = self.tx.send(Msg::HatPedal(false));
                }
                14 => {
                    // The pad's `freq` is the pitch of the A key.
                    let base = self.pads[self.pad_sel].freq;
                    let f = base * 2f32.powf(k as f32 / 12.0 + self.octave as f32);
                    self.held_notes.push((k, self.pad_sel, f));
                    let _ = self.tx.send(Msg::Note(self.pad_sel, f));
                }
                15 => self.octave = (self.octave - 1).max(-3),
                16 => self.octave = (self.octave + 1).min(3),
                // Shift+melody key held = roll that note (a pan's
                // sustain); release stops it.
                17 => {
                    let base = self.pads[self.pad_sel].freq;
                    let f = base * 2f32.powf(k as f32 / 12.0 + self.octave as f32);
                    self.note_rolls.push((k, self.pad_sel, f));
                    let _ = self.tx.send(Msg::NoteRoll(self.pad_sel, f, true));
                }
                // Banks: number keys pick one; Tab flips between the two
                // kits, which mirror each other slot for slot.
                19 => {
                    let next = if self.bank == Bank::Physical {
                        Bank::Modal
                    } else {
                        Bank::Physical
                    };
                    self.select_bank(next);
                }
                20 => self.select_bank(BANKS[k]),
                18 => {
                    let mut kept = Vec::new();
                    for (semi, pad, f) in self.note_rolls.drain(..) {
                        if semi == k {
                            let _ = self.tx.send(Msg::NoteRoll(pad, f, false));
                        } else {
                            kept.push((semi, pad, f));
                        }
                    }
                    self.note_rolls = kept;
                    let mut kept = Vec::new();
                    for (semi, pad, f) in self.held_notes.drain(..) {
                        if semi == k {
                            if !self.sustain {
                                let _ = self.tx.send(Msg::NoteOff(pad, f));
                            }
                        } else {
                            kept.push((semi, pad, f));
                        }
                    }
                    self.held_notes = kept;
                }
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
            ui.label("loop");
            let state = match self.loop_ctl.state.load(Relaxed) {
                1 => "recording",
                2 => "playing",
                3 => "overdub",
                4 => "stopped",
                _ => "idle",
            };
            let layers = self.loop_ctl.layers.load(Relaxed);
            let len = self.loop_ctl.len.get();
            ui.horizontal(|ui| {
                let label = match state {
                    "idle" => "record",
                    "recording" => "close & play",
                    "playing" => "overdub",
                    "overdub" => "jam",
                    _ => "play",
                };
                if ui.button(label).clicked() {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Toggle));
                }
                if matches!(state, "playing" | "overdub") && ui.small_button("stop").clicked() {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Stop));
                }
                if ui.small_button("undo").clicked() {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Undo));
                }
                if ui.small_button("clear").clicked() {
                    let _ = self.tx.send(Msg::Loop(LoopCmd::Clear));
                }
            });
            let text = if state == "recording" && self.loop_ctl.tap.load(Relaxed) {
                format!(
                    "tapping… {:.0} bpm (space to set it)",
                    self.loop_ctl.bpm.get()
                )
            } else if state == "recording" {
                format!("recording… {:.1}s", self.loop_ctl.pos.get())
            } else if len > 0.0 {
                format!(
                    "{state} · {layers} layer{} · {len:.1}s · {:.0} bpm",
                    if layers == 1 { "" } else { "s" },
                    self.loop_ctl.bpm.get()
                )
            } else {
                "idle — space to record".into()
            };
            let cycling = matches!(state, "playing" | "overdub");
            let bar = egui::ProgressBar::new(if cycling {
                self.loop_ctl.pos.get()
            } else {
                0.0
            })
            .text(text);
            // Red while overdubbing: the pedal's light.
            let bar = if state == "overdub" {
                bar.fill(egui::Color32::from_rgb(180, 60, 60))
            } else {
                bar
            };
            ui.add(bar);
            ui.horizontal(|ui| {
                let mut tap = self.loop_ctl.tap.load(Relaxed);
                if ui
                    .toggle_value(&mut tap, "tap tempo")
                    .on_hover_text(
                        "arm, then space, tap one pad steadily (4+ times), space: \
                         the tempo is set from your taps and the pad keeps tapping",
                    )
                    .changed()
                {
                    self.loop_ctl.tap.store(tap, Relaxed);
                }
                let bpm = self.loop_ctl.bpm.get();
                if ui.small_button("÷2").clicked() {
                    self.loop_ctl.bpm.set((bpm * 0.5).max(20.0));
                }
                if ui.small_button("×2").clicked() {
                    self.loop_ctl.bpm.set((bpm * 2.0).min(400.0));
                }
            });
            ui.horizontal(|ui| {
                let mut q = self.loop_ctl.quantize.load(Relaxed);
                if ui.checkbox(&mut q, "quantize").changed() {
                    self.loop_ctl.quantize.store(q, Relaxed);
                }
                let mut c = self.loop_ctl.click.load(Relaxed);
                if ui.checkbox(&mut c, "click").changed() {
                    self.loop_ctl.click.store(c, Relaxed);
                }
            });
            ctl_slider(ui, &self.loop_ctl.bpm, 20.0..=400.0, false, "bpm");
            ui.separator();
            ui.small("space — loop: record / close & play / overdub ↔ jam");
            ui.small("shift+space — stop / restart");
            ui.small("backspace — undo last layer (shift: clear)");
            ui.label("bank");
            ui.horizontal_wrapped(|ui| {
                for (k, b) in BANKS.iter().enumerate() {
                    let label = format!("{} {}", k + 1, b.name());
                    if ui.selectable_label(self.bank == *b, label).clicked() {
                        self.select_bank(*b);
                    }
                }
            });
            let row: Vec<String> = {
                let slots = self.slots();
                ROW_KEYS
                    .iter()
                    .zip(slots.iter())
                    .map(|(key, s)| {
                        let what = match s {
                            Slot::None => "—".to_string(),
                            Slot::Pad(i) => self.pads[*i].name.to_string(),
                            Slot::Mesh(i) => self.mesh_pads[*i].0.to_string(),
                            Slot::Cymbal(i) => self.cymbal_pads[*i].0.to_string(),
                            Slot::HatClosed => "hat".into(),
                            Slot::HatOpen => "open hat".into(),
                            Slot::Foot => "foot".into(),
                        };
                        format!("{}:{}", key_label(*key), what)
                    })
                    .collect()
            };
            match self.bank {
                Bank::Tuned(_) => ui.small(
                    "melody keys: home row white, W E T Y U O P black, [ ] octave, shift = roll",
                ),
                Bank::Physical => ui.small(format!("{}  (shift = hard hit)", row.join(" "))),
                _ => ui.small(format!("{}  (shift = roll)", row.join(" "))),
            };
            ui.small("1–8 — bank · tab — modal ↔ physical");
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
                    // The pads the keyboard is playing right now, in key
                    // order (the bank), labelled with their keys; every
                    // other modal pad is a fold away.
                    let mut strikes = Vec::new();
                    let slots = self.slots();
                    let on_row: Vec<usize> = slots
                        .iter()
                        .filter_map(|s| match s {
                            Slot::Pad(i) => Some(*i),
                            _ => None,
                        })
                        .collect();
                    let tuned = match self.bank {
                        Bank::Tuned(n) => self.pad_named(n),
                        _ => None,
                    };
                    ui.horizontal_wrapped(|ui| {
                        if let Some(i) = tuned {
                            let label = format!("{} (melody keys)", self.pads[i].name);
                            if ui.selectable_label(self.pad_sel == i, label).clicked() {
                                strikes.push(i);
                            }
                        }
                        for (k, s) in slots.iter().enumerate() {
                            if let Slot::Pad(i) = s {
                                let label = format!("{} {}", key_label(ROW_KEYS[k]), self.pads[*i].name);
                                if ui.selectable_label(self.pad_sel == *i, label).clicked() {
                                    strikes.push(*i);
                                }
                            }
                        }
                        if on_row.is_empty() && tuned.is_none() {
                            ui.small("bank 2 is the physical kit — see its panels");
                        }
                    });
                    egui::CollapsingHeader::new("all modal pads")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for (i, p) in self.pads.iter().enumerate() {
                                    if on_row.contains(&i) || tuned == Some(i) {
                                        continue;
                                    }
                                    if ui.selectable_label(self.pad_sel == i, p.name).clicked() {
                                        strikes.push(i);
                                    }
                                }
                            });
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
                                ModeSet::Gong,
                                ModeSet::Hat,
                                ModeSet::Snare,
                                ModeSet::Marimba,
                                ModeSet::Vibes,
                                ModeSet::Pan,
                                ModeSet::Cowbell,
                                ModeSet::Woodblock,
                                ModeSet::Tambourine,
                                ModeSet::Simmons,
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
                    edited |= slider(ui, &mut p.bloom_delay, 0.005..=1.0, true, "bloom delay");
                    edited |= slider(ui, &mut p.bloom_spread, 0.0..=1.0, false, "bloom spread");
                    edited |= slider(ui, &mut p.drive, 0.0..=6.0, false, "drive");
                    edited |= slider(ui, &mut p.shimmer, 0.0..=1.0, false, "shimmer");
                    edited |= slider(ui, &mut p.attack, 0.0..=0.02, false, "mallet (attack s)");
                    edited |= slider(ui, &mut p.tremolo, 0.0..=1.0, false, "tremolo");
                    edited |= slider(
                        ui,
                        &mut p.bloom_lo,
                        0.0..=1.0,
                        false,
                        "bloom aim (top↔harmonics)",
                    );
                    edited |= slider(ui, &mut p.bleed, 0.0..=0.6, false, "neighbours (bleed)");
                    edited |= slider(ui, &mut p.roll_rate, 4.0..=30.0, false, "roll rate (/s)");
                    edited |= slider(ui, &mut p.roll_strength, 0.1..=1.0, false, "roll strength");
                    ui.spacing_mut().slider_width = track_width(ui);
                    edited |= ui
                        .add(egui::Slider::new(&mut p.claps, 1..=6).text("claps"))
                        .changed();
                    edited |= slider(ui, &mut p.level, 0.0..=1.0, false, "level");
                    let mut chokes = p.choke.is_some();
                    if ui.checkbox(&mut chokes, "choke group").changed() {
                        p.choke = if chokes { Some(0) } else { None };
                        edited = true;
                    }
                    ui.checkbox(&mut self.melody, "melody keys play this pad");
                    if self.melody {
                        ui.horizontal(|ui| {
                            edited |= ui.checkbox(&mut p.damper, "damper on release").changed();
                            ui.checkbox(&mut self.sustain, "sustain pedal");
                        });
                        ui.small(format!(
                            "A S D F G H J K L ; ' white · W E T Y U O P black · [ ] octave ({:+}) · shift = roll",
                            self.octave
                        ));
                    }
                    // Ship changed params to the ringing pad — knob moves
                    // land on sounds already in the air.
                    if edited {
                        let _ = self.tx.send(Msg::Pad(self.pad_sel, p.to_pad()));
                    }
                    // The mode ladder: this pad's partials, lit as they ring.
                    self.view_ctl.pad.store(self.pad_sel, Relaxed);
                    if let Ok(v) = self.view.lock() {
                        draw_ladder(ui, &v.ladder, v.noise, &mut self.gain_ladder);
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
                    edited |= slider(ui, &mut m.decay, 0.02..=3.0, true, "decay");
                    edited |= slider(ui, &mut m.hf_damp, 0.0..=0.95, false, "overtone damp");
                    edited |= slider(ui, &mut m.tension, 0.0..=80.0, false, "tension");
                    edited |= slider(ui, &mut m.strike_pos, 0.0..=1.0, false, "strike pos");
                    edited |= slider(ui, &mut m.rattle, 0.0..=1.0, false, "rattle");
                    edited |= slider(ui, &mut m.click, 0.0..=1.0, false, "beater click");
                    edited |= slider(ui, &mut m.drive, 0.0..=6.0, false, "drive");
                    edited |= slider(ui, &mut m.air, 0.0..=1.5, false, "shell air");
                    edited |= slider(ui, &mut m.reso_freq, 25.0..=300.0, true, "reso head");
                    edited |= slider(ui, &mut m.reso_decay, 0.05..=3.0, true, "reso decay");
                    edited |= ui
                        .checkbox(&mut m.two_heads, "two heads (full resonant membrane)")
                        .changed();
                    edited |= slider(ui, &mut m.shell, 0.0..=0.3, false, "shell coupling");
                    edited |= slider(ui, &mut m.hardness, 0.0..=1.0, false, "stick hardness");
                    edited |= slider(ui, &mut m.mallet, 0.8..=5.0, false, "mallet size");
                    edited |= slider(ui, &mut m.level, 0.0..=4.0, false, "level");
                    if edited {
                        let _ = self.tx.send(Msg::MeshPad(self.mesh_sel, *m));
                    }
                    // The head itself, moving.
                    self.view_ctl.mesh.store(self.mesh_sel, Relaxed);
                    let mut energy = self.view_ctl.energy.load(Relaxed);
                    if ui
                        .checkbox(&mut energy, "energy view (mode shapes, no flicker)")
                        .changed()
                    {
                        self.view_ctl.energy.store(energy, Relaxed);
                    }
                    if let Ok(v) = self.view.lock() {
                        let w = mesh::Mesh::width();
                        let size = ui.available_width().min(220.0);
                        let field = if energy { &v.mesh_energy } else { &v.mesh };
                        draw_surface(ui, field, w, size, &mut self.gain_mesh, energy);
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
                    // The plate itself, bending.
                    self.view_ctl.cymbal.store(self.cymbal_sel, Relaxed);
                    if let Ok(v) = self.view.lock() {
                        let w = cymbal::Cymbal::width();
                        let size = ui.available_width().min(220.0);
                        let energy = self.view_ctl.energy.load(Relaxed);
                        let field = if energy { &v.cymbal_energy } else { &v.cymbal };
                        draw_surface(ui, field, w, size, &mut self.gain_cymbal, energy);
                    }

                    // ---- Hi-hat: two plates on a pedal.
                    ui.separator();
                    ui.heading("hi-hat · two plates");
                    ui.horizontal(|ui| {
                        if ui.button("closed").clicked() {
                            if !self.pedal_down {
                                let _ = self.tx.send(Msg::HatPedal(true));
                            }
                            let _ = self.tx.send(Msg::HatStrike(0.9));
                        }
                        if ui.button("open").clicked() {
                            if !self.pedal_down {
                                let _ = self.tx.send(Msg::HatPedal(false));
                            }
                            let _ = self.tx.send(Msg::HatStrike(0.9));
                        }
                        let mut down = self.pedal_down;
                        if ui.toggle_value(&mut down, "pedal").changed() {
                            self.pedal_down = down;
                            let _ = self.tx.send(Msg::HatPedal(down));
                        }
                    });
                    let h = &mut self.hat;
                    let mut edited = false;
                    edited |= slider(ui, &mut h.plate.stiffness, 0.0..=1.0, false, "stiffness");
                    edited |= slider(ui, &mut h.plate.dome, 10.0..=600.0, true, "dome");
                    edited |= slider(ui, &mut h.plate.decay, 0.1..=8.0, true, "decay");
                    edited |= slider(ui, &mut h.plate.hf_damp, 0.0..=1.0, false, "hf damp");
                    edited |= slider(ui, &mut h.gap, 0.02..=1.0, true, "gap");
                    edited |= slider(ui, &mut h.press, 0.0..=1.0, false, "press");
                    edited |= slider(ui, &mut h.click, 0.0..=2.0, false, "stick click");
                    edited |= slider(ui, &mut h.level, 0.0..=4.0, false, "level");
                    if edited {
                        let _ = self.tx.send(Msg::HatPad(*h));
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
    for n in MODAL_ROW.iter().chain(PERC_ROW.iter()) {
        assert!(
            n.is_empty() || pads.iter().any(|p| p.name == *n),
            "no modal pad named {n}"
        );
    }
    for b in BANKS {
        if let Bank::Tuned(n) = b {
            assert!(pads.iter().any(|p| p.name == n), "no tuned pad named {n}");
        }
    }
    let mesh_pads = mesh::default_kit();
    assert!(
        mesh_pads.len() >= 5,
        "the physical bank wants five mesh pads"
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
    let loop_ctl = Arc::new(LoopCtl {
        bpm: AtomicF32::new(100.0),
        quantize: AtomicBool::new(false),
        click: AtomicBool::new(false),
        tap: AtomicBool::new(false),
        state: AtomicUsize::new(0),
        layers: AtomicUsize::new(0),
        pos: AtomicF32::new(0.0),
        len: AtomicF32::new(0.0),
    });
    let view = Arc::new(Mutex::new(ViewData {
        mesh: vec![0.0; mesh::Mesh::width() * mesh::Mesh::width()],
        cymbal: vec![0.0; cymbal::Cymbal::width() * cymbal::Cymbal::width()],
        ladder: Vec::new(),
        noise: (0.0, 0.0, 1500.0),
        mesh_energy: vec![0.0; mesh::Mesh::width() * mesh::Mesh::width()],
        cymbal_energy: vec![0.0; cymbal::Cymbal::width() * cymbal::Cymbal::width()],
    }));
    let view_ctl = Arc::new(ViewCtl {
        pad: AtomicUsize::new(0),
        mesh: AtomicUsize::new(0),
        cymbal: AtomicUsize::new(0),
        energy: AtomicBool::new(true),
    });
    let cymbal_pads = cymbal::default_kit();
    for n in ["crash", "ride"] {
        assert!(
            cymbal_pads.iter().any(|(c, _)| *c == n),
            "no cymbal named {n}"
        );
    }
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
        loop_ctl.clone(),
        view.clone(),
        view_ctl.clone(),
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
        hat: hihat::default_params(),
        pedal_down: false,
        melody: false,
        octave: 0,
        note_rolls: Vec::new(),
        held_notes: Vec::new(),
        sustain: false,
        loop_ctl,
        view,
        view_ctl,
        gain_mesh: 1e-3,
        gain_cymbal: 1e-3,
        gain_ladder: 1e-3,
        bank: Bank::Modal,
        row_down: [false; NROW],
        row_rolling: [None; NROW],
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
