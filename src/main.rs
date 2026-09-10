//! Offline demo render: the drum kit voice by voice, the same sung note
//! becoming five different words, then everyone together. Writes out.wav.
//! For the realtime desk, `cargo run --release --bin desk`.

use timber::modal::{self, Hit};
use timber::string::{self, Pluck};
use timber::util::{Rng, SAMPLE_RATE, SR, place};
use timber::voice::{self, Note};

fn main() {
    let mut rng = Rng(0x74696d62); // "timb"
    let mut mix = Vec::new();

    // ---- The kit: one modal engine, five mode lists ----------------------
    let kick = Hit {
        freq: 50.0,
        modes: modal::MEMBRANE_CENTER,
        duration: 0.8,
        glide: 0.9, // the deep 808-ish drop is pure tension modulation
        glide_time: 0.06,
        level: 0.95,
        ..Default::default()
    };
    let tom = |freq| Hit {
        freq,
        duration: 1.2,
        glide: 0.25,
        glide_time: 0.12,
        level: 0.8,
        ..Default::default()
    };
    let snare = Hit {
        freq: 185.0,
        duration: 0.5,
        glide: 0.15,
        glide_time: 0.05,
        noise: 0.9,
        noise_decay: 0.09,
        level: 0.75,
        ..Default::default()
    };
    // A hat is nearly all rattle: no modes, just a fast bright burst.
    let hat = |open: bool| Hit {
        modes: &[],
        duration: if open { 0.5 } else { 0.08 },
        noise: 1.0,
        noise_decay: if open { 0.18 } else { 0.025 },
        level: 0.3,
        ..Default::default()
    };
    let bell = Hit {
        freq: 440.0,
        modes: modal::BELL,
        duration: 5.0,
        level: 0.5,
        ..Default::default()
    };

    // ---- Scene 1: the kit, voice by voice --------------------------------
    for (t, h) in [
        (0.0, kick),
        (0.8, tom(110.0)), // listen for the pitch droop
        (1.7, snare),
        (2.4, hat(false)),
        (2.6, hat(true)),
        (3.4, bell),
    ] {
        place(&mut mix, &modal::render(&h, &mut rng), t);
    }

    // ---- Scene 2: one pitch, five words ----------------------------------
    // Every note here is the same E. Only the formant motion changes:
    // this is the difference between beeps and voicing.
    let syllable = |from, to| Note {
        freq: 165.0,
        duration: 0.55,
        from,
        to,
        ..Default::default()
    };
    let words = [
        syllable(voice::AH, voice::AH), // "ah"
        syllable(voice::EE, voice::AH), // "yah"
        syllable(voice::OO, voice::AH), // "wah"
        syllable(voice::EE, voice::EH), // "yeh"
        syllable(voice::OO, voice::OH), // "woh"
    ];
    for (i, w) in words.iter().enumerate() {
        place(&mut mix, &voice::render(w, &mut rng), 6.5 + i as f32 * 0.7);
    }

    // ---- Scene 3: groove + string riff + a sung phrase -------------------
    let bpm = 96.0;
    let beat = 60.0 / bpm;
    let t0 = 11.0;

    for bar in 0..2 {
        let bt = t0 + bar as f32 * 4.0 * beat;
        place(&mut mix, &modal::render(&kick, &mut rng), bt);
        place(&mut mix, &modal::render(&kick, &mut rng), bt + 2.0 * beat);
        place(&mut mix, &modal::render(&snare, &mut rng), bt + beat);
        place(&mut mix, &modal::render(&snare, &mut rng), bt + 3.0 * beat);
        for i in 0..8 {
            let open = i == 7;
            place(
                &mut mix,
                &modal::render(&hat(open), &mut rng),
                bt + i as f32 * beat / 2.0,
            );
        }
    }
    // Tom fill rolling into the end of bar 2.
    for (i, f) in [150.0, 120.0, 95.0].iter().enumerate() {
        place(
            &mut mix,
            &modal::render(&tom(*f), &mut rng),
            t0 + (7.25 + 0.25 * i as f32) * beat,
        );
    }

    // Steel-string riff, A minor pentatonic, eighth notes.
    let steel = Pluck {
        damping: 0.1,
        decay: 0.998,
        pluck_pos: 0.12,
        level: 0.45,
        duration: 1.5,
        ..Default::default()
    };
    let scale = [110.0, 130.81, 146.83, 164.81, 196.0, 220.0];
    let riff = [5, 3, 4, 2, 3, 1, 2, 0];
    for bar in 0..2 {
        let bt = t0 + bar as f32 * 4.0 * beat;
        for (i, &deg) in riff.iter().enumerate() {
            let p = Pluck {
                freq: scale[deg],
                ..steel
            };
            place(
                &mut mix,
                &string::render(&p, &mut rng),
                bt + i as f32 * beat / 2.0,
            );
        }
    }

    // The voice takes bar 2: yah — woh — wee — aah (long, with vibrato).
    let sung = |freq, duration, from, to| Note {
        freq,
        duration,
        level: 0.7,
        ..Note {
            from,
            to,
            ..Default::default()
        }
    };
    let vt = t0 + 4.0 * beat;
    place(
        &mut mix,
        &voice::render(&sung(220.0, beat * 0.9, voice::EE, voice::AH), &mut rng),
        vt,
    );
    place(
        &mut mix,
        &voice::render(&sung(196.0, beat * 0.9, voice::OO, voice::OH), &mut rng),
        vt + beat,
    );
    place(
        &mut mix,
        &voice::render(&sung(164.81, beat * 0.9, voice::OO, voice::EE), &mut rng),
        vt + 2.0 * beat,
    );
    place(
        &mut mix,
        &voice::render(&sung(220.0, beat * 3.5, voice::AH, voice::AH), &mut rng),
        vt + 3.0 * beat,
    );

    // ---- Finale: bell, low stiff string, one last kick -------------------
    let tf = t0 + 8.0 * beat;
    place(&mut mix, &modal::render(&kick, &mut rng), tf);
    place(
        &mut mix,
        &modal::render(
            &Hit {
                freq: 220.0,
                ..bell
            },
            &mut rng,
        ),
        tf,
    );
    let low = Pluck {
        freq: 55.0,
        duration: 6.0,
        decay: 0.999,
        damping: 0.35,
        stiffness: 0.45,
        pluck_pos: 0.08,
        ..Default::default()
    };
    place(&mut mix, &string::render(&low, &mut rng), tf);

    // ---- Normalize and write ---------------------------------------------
    let peak = mix.iter().fold(0.0f32, |m, s| m.max(s.abs())).max(1e-9);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create("out.wav", spec).expect("create out.wav");
    for s in &mix {
        wav.write_sample((s / peak * 0.9 * i16::MAX as f32) as i16)
            .expect("write sample");
    }
    wav.finalize().expect("finalize wav");
    println!("wrote out.wav ({:.1}s)", mix.len() as f32 / SR);
}
