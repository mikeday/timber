# timber

A physical-modeling synthesis playground in Rust, built on one idea:
timbre is the essence of music, not something sprayed onto notes and
rhythms afterwards. So every sound here starts from a *mechanism* —
something struck, plucked, bowed or blown — and the timbre is whatever
that mechanism does.

## What's in it

Models, in `src/`:

- `string` / `stream` — Karplus-Strong and a two-line digital
  waveguide string with a stick-slip bow, sympathetic coupling and a
  drone mode.
- `modal` / `drums` — a modal kit: resonator banks with typed-in mode
  tables (membranes, bell, triangle, and synth cymbal/ride/gong/hi-hat
  pictures), plus rattle, shimmer, drive and a wash bloom. The
  "record" drums.
- `mesh` — drums as drums: a 2D finite-difference membrane with a
  Hertz-contact stick, tension pitch-bend, an air cavity and resonant
  head, snare wires, calibrated against recordings (`samples/`). The
  "instrument" drums.
- `plate` / `cymbal` — the same disc with bending: a finite-difference
  stiff plate, and the modal von Kármán cymbal the desk plays (mode
  shapes from the grid, cubic coupling between them).
- `hihat` — two modal cymbals on a pedal, meeting through contact
  springs: the choke and the chick are the physics of bronze touching.
- `stick` / `grid` — shared by the struck models: the Hertz-contact
  stick, and the disc every membrane and plate lives on.
- `voice` / `mouth` / `tract` / `speak` / `sing` — a formant voice, and
  a Kelly-Lochbaum vocal tract with a nasal branch and a tongue tip,
  a phoneme sequencer with coarticulation, and a score for
  "Daisy Bell".
- `body` — resonator-bank bodies (guitar, violin, shell, plate,
  cathedral) on the mixer strips.
- `looper` — an event looper: records hits on the audio thread's
  clock and replays them into the live instruments, so every knob
  stays live across the loop.
- `analyze` — measurement: envelope, brightness, pitch bend, mode
  tables with decays, for our sounds and for recordings.

## Running it

    cargo run --release --bin desk

The desk is a realtime mixing desk (egui + cpal). Keys: the number
row `1`–`=` plays the modal kit (shift = roll); the bottom row plays
the physical kit — `Z X C V B` mesh drums, `N M` cymbals (shift =
hard hit), `,` the hi-hat with `.` held as the pedal; `A S D F G H J
K` pluck strings, `Q W E R T` vowels. Hold
the bow surface to bow, the mouth surface to sing; type phonemes in
the tract box to speak. `Space` runs the looper (record → play →
overdub ↔ jam; shift+Space stops), `Backspace` undoes a layer
(shift+Backspace clears).

`cargo run --release --bin timber` renders an offline demo to
`out.wav`. `cargo test --release` runs the tests, most of which
measure the sound (pitch, decay, formants, wash) rather than the code.

Diagnostics in `examples/` print what the models do — `compare`
(modal vs mesh drum), `analyze` (a recording, or `mesh:tomlo`,
`cymbal:crash`, `hat:open`), `stickprobe`, `washprobe`, `doublehit`,
`modalprobe`, `hatprobe` and friends — and were how most of the
calibration decisions were made.

One honest limit: the physical cymbals and hi-hat run on a disc whose
modes stop near 3 kHz, so they come out darker than the metal they
model — a crash more like a small gong, a hat without its top. The
mechanisms (the wash arriving after the hit, the pedal, the choke)
are right; the synth pictures in the modal kit have the brightness. `samples/` holds CC0 single hits from freesound.org used to
calibrate the mesh kit; `samples/README` credits them.
