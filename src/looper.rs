//! An event looper: records what you *did* — which pad, when, how
//! hard — on the audio thread's sample clock, and replays it into the
//! live instruments. Looping the events rather than the audio keeps
//! every knob live across the loop: record a bar of kick and snare,
//! then retune the kick or swap the snare for a gong while it cycles.
//!
//! Tap to start recording (the first hit is beat one), tap to close —
//! the loop is as long as you played — and it plays at once. Playing
//! is *jamming*: what you play over it sounds but is not kept. Tap
//! again to overdub — now a pass that added something becomes a layer
//! — and again to go back to jamming; layers undo in order. Stop is a
//! separate gesture, so neither stopping nor recording can happen by
//! accident. (The convention loop pedals settled on.) An optional grid
//! snaps hits to sixteenths and the loop length to bars at a tempo.
//!
//! Generic over the event type so the desk can loop its own messages.

use crate::util::SR;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Idle,
    Recording,
    /// Cycling; what you play sounds but is not kept.
    Playing,
    /// Cycling and keeping what you play, a layer per pass.
    Overdub,
    Stopped,
}

pub struct Looper<E> {
    state: State,
    /// Sample clock at the loop's origin (beat one).
    origin: u64,
    /// Loop length in samples once closed.
    len: u64,
    /// Committed layers, each a list of (offset, event) in order.
    layers: Vec<Vec<(u64, E)>>,
    /// Events recorded during the current pass, committed on wrap.
    pending: Vec<(u64, E)>,
    /// All committed events merged and sorted, with a play cursor.
    merged: Vec<(u64, E)>,
    cursor: usize,
    last_pos: u64,
    /// Grid: beats per minute and whether to snap to it.
    pub bpm: f32,
    pub quantize: bool,
    /// Tap tempo: the next recording is a steady tap on one pad, and
    /// closing it sets the tempo from the taps and replaces what was
    /// played with a bar of four exact taps that keep going as the
    /// click. The estimate follows the *recent* taps (a weighted
    /// average leaning on the last few), so speeding up or slowing
    /// down while tapping moves it, and it is readable while tapping.
    pub tap: bool,
    /// The running tap estimate, samples per tap.
    tap_est: Option<f32>,
}

impl<E: Clone> Looper<E> {
    pub fn new() -> Self {
        Looper {
            state: State::Idle,
            origin: 0,
            len: 0,
            layers: Vec::new(),
            pending: Vec::new(),
            merged: Vec::new(),
            cursor: 0,
            last_pos: 0,
            bpm: 100.0,
            quantize: false,
            tap: false,
            tap_est: None,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Loop length in seconds (0 until closed).
    pub fn len_secs(&self) -> f32 {
        self.len as f32 / SR
    }

    /// Playhead as a fraction of the loop, for display.
    pub fn position(&self, now: u64) -> f32 {
        match self.state {
            State::Recording => ((now - self.origin) as f32 / SR).max(0.0),
            State::Playing | State::Overdub if self.len > 0 => {
                ((now - self.origin) % self.len) as f32 / self.len as f32
            }
            _ => 0.0,
        }
    }

    /// The tempo the taps so far imply, while tapping.
    pub fn tap_bpm(&self) -> Option<f32> {
        self.tap_est.map(|t| 60.0 * SR / t)
    }

    /// Re-estimate from the taps recorded so far: a weighted average
    /// of the last five gaps, newest weighted most (0.55 per step
    /// back), each gap clamped to within a factor of two of the
    /// running estimate so a doubled or missed tap can't throw it.
    fn estimate_taps(&mut self) {
        let offs: Vec<u64> = self.pending.iter().map(|(o, _)| *o).collect();
        if offs.len() < 2 {
            return;
        }
        let mut est = self.tap_est;
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let gaps: Vec<f32> = offs.windows(2).map(|w| (w[1] - w[0]) as f32).collect();
        for (k, &g) in gaps.iter().rev().take(5).enumerate() {
            let g = match est {
                Some(e) => g.clamp(0.5 * e, 2.0 * e),
                None => g,
            };
            let w = 0.55f32.powi(k as i32);
            num += w * g;
            den += w;
            if est.is_none() {
                est = Some(g);
            }
        }
        self.tap_est = Some(num / den);
    }

    fn sixteenth(&self) -> u64 {
        ((SR * 60.0 / self.bpm.max(20.0)) / 4.0).round() as u64
    }

    fn snap(&self, offset: u64) -> u64 {
        if !self.quantize {
            return offset;
        }
        let g = self.sixteenth();
        ((offset + g / 2) / g) * g
    }

    /// The one-button control: idle → recording → playing, then
    /// playing ↔ overdub. (Stopped resumes to playing.) Returns true
    /// when closing a tap-tempo recording set the tempo.
    pub fn toggle(&mut self, now: u64) -> bool {
        let mut tapped = false;
        self.state = match self.state {
            State::Idle => {
                self.origin = now;
                self.layers.clear();
                self.pending.clear();
                self.merged.clear();
                self.tap_est = None;
                State::Recording
            }
            State::Recording if self.tap && self.tap_est.is_some() => {
                // Tap tempo: the running estimate is the tempo; the
                // loop is four taps of it, exactly spaced, and beat one
                // is the *last* tap — the next click lands one beat
                // after the last thing played, whatever the tempo did
                // along the way (anchored to the first tap, a drifting
                // tempo left the clicks out of phase with the hand).
                let t = self.tap_est.unwrap_or(SR).max(1.0);
                self.bpm = (60.0 * SR / t).clamp(20.0, 400.0);
                self.quantize = true;
                let t = (60.0 * SR / self.bpm).round() as u64;
                let last = self.pending.iter().map(|(o, _)| *o).max().unwrap_or(0);
                self.origin += last;
                let e = self.pending[0].1.clone();
                self.pending = (0..4).map(|k| (k * t, e.clone())).collect();
                self.len = 4 * t;
                self.tap = false;
                tapped = true;
                self.commit();
                self.start_at(now);
                State::Playing
            }
            State::Recording => {
                let mut len = (now - self.origin).max(1);
                if self.quantize {
                    // Whole bars, at least one.
                    let bar = 16 * self.sixteenth();
                    len = ((len + bar / 2) / bar).max(1) * bar;
                }
                self.len = len;
                self.tap = false;
                self.commit();
                self.start_at(now);
                State::Playing
            }
            State::Playing => State::Overdub,
            State::Overdub => {
                // Leaving overdub keeps what this pass added.
                self.commit();
                State::Playing
            }
            State::Stopped => {
                // Resume from beat one.
                self.origin = now;
                self.cursor = 0;
                self.last_pos = 0;
                State::Playing
            }
        };
        tapped
    }

    /// Stop a cycling loop (keeping it), or restart a stopped one from
    /// beat one. Nothing to stop while idle or recording.
    pub fn stop(&mut self, now: u64) {
        match self.state {
            State::Playing | State::Overdub => {
                self.commit();
                self.state = State::Stopped;
            }
            State::Stopped => {
                self.origin = now;
                self.cursor = 0;
                self.last_pos = 0;
                self.state = State::Playing;
            }
            _ => {}
        }
    }

    /// Record an event played now. Only recording and overdub keep it.
    pub fn record(&mut self, now: u64, e: E) {
        match self.state {
            State::Recording => {
                // The first hit is beat one: the loop starts there,
                // not at the button press.
                if self.pending.is_empty() {
                    self.origin = now;
                }
                // Taps are timed raw: the grid is what they will set.
                let raw = now.saturating_sub(self.origin);
                let off = if self.tap { raw } else { self.snap(raw) };
                self.pending.push((off, e));
                if self.tap {
                    self.estimate_taps();
                }
            }
            State::Overdub => {
                let off = self.snap((now - self.origin) % self.len) % self.len;
                self.pending.push((off, e));
            }
            _ => {}
        }
    }

    /// Begin playing at the loop's current phase: events already
    /// behind `now` in this pass are skipped, not fired late (closing
    /// a tap loop a beat after the last tap must not click at once).
    fn start_at(&mut self, now: u64) {
        self.last_pos = (now - self.origin) % self.len.max(1);
        self.cursor = self
            .merged
            .iter()
            .position(|(o, _)| *o >= self.last_pos)
            .unwrap_or(self.merged.len());
    }

    /// Turn the pending pass into a layer.
    fn commit(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut layer = std::mem::take(&mut self.pending);
        for (off, _) in &mut layer {
            *off %= self.len.max(1);
        }
        layer.sort_by_key(|(o, _)| *o);
        self.layers.push(layer);
        self.rebuild();
    }

    fn rebuild(&mut self) {
        self.merged = self.layers.iter().flatten().cloned().collect();
        self.merged.sort_by_key(|(o, _)| *o);
        // Keep the cursor pointing at the first event not yet due at
        // the current position.
        self.cursor = self
            .merged
            .iter()
            .position(|(o, _)| *o > self.last_pos)
            .unwrap_or(self.merged.len());
    }

    /// Drop the most recent layer.
    pub fn undo(&mut self) {
        self.pending.clear();
        self.layers.pop();
        self.rebuild();
    }

    pub fn clear(&mut self) {
        self.state = State::Idle;
        self.layers.clear();
        self.pending.clear();
        self.merged.clear();
        self.cursor = 0;
        self.len = 0;
    }

    /// Advance to sample `now`: returns the events due at this sample
    /// (the loop's are sample-exact). Call once per output frame.
    pub fn due(&mut self, now: u64, out: &mut Vec<E>) {
        if !matches!(self.state, State::Playing | State::Overdub) || self.len == 0 {
            return;
        }
        let pos = (now - self.origin) % self.len;
        if pos < self.last_pos {
            // Wrapped: commit the pass, start again from the top.
            self.commit();
            self.cursor = 0;
        }
        self.last_pos = pos;
        while self.cursor < self.merged.len() && self.merged[self.cursor].0 <= pos {
            out.push(self.merged[self.cursor].1.clone());
            self.cursor += 1;
        }
    }

    /// Is `now` on a beat (for a metronome), and is it beat one?
    pub fn beat(&self, now: u64) -> Option<bool> {
        if self.state == State::Idle || self.state == State::Stopped {
            return None;
        }
        let beat = 4 * self.sixteenth();
        let t = now - self.origin;
        if t.is_multiple_of(beat) {
            Some(t.is_multiple_of(4 * beat))
        } else {
            None
        }
    }
}

impl<E: Clone> Default for Looper<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(l: &mut Looper<u8>, from: u64, to: u64) -> Vec<(u64, u8)> {
        let mut got = Vec::new();
        let mut out = Vec::new();
        for t in from..to {
            out.clear();
            l.due(t, &mut out);
            for e in &out {
                got.push((t, *e));
            }
        }
        got
    }

    #[test]
    fn records_closes_and_replays_in_time() {
        let mut l: Looper<u8> = Looper::new();
        l.toggle(1000); // record
        l.record(1000, 1);
        l.record(1500, 2);
        l.toggle(2000); // close: len 1000
        assert_eq!(l.state(), State::Playing);
        assert_eq!(l.layers(), 1);
        let got = play(&mut l, 2000, 4000);
        assert_eq!(got, vec![(2000, 1), (2500, 2), (3000, 1), (3500, 2)]);
    }

    #[test]
    fn overdub_lands_on_the_next_pass_and_undoes() {
        let mut l: Looper<u8> = Looper::new();
        l.toggle(0);
        l.record(0, 1);
        l.toggle(1000);
        let _ = play(&mut l, 1000, 1200);
        l.record(1100, 9); // jamming: sounds, not kept
        l.toggle(1200); // overdub
        l.record(1250, 7); // played live mid-pass
        let got = play(&mut l, 1200, 3000);
        // Not in this pass (it sounded live), in the next two.
        assert_eq!(
            got,
            vec![(2000, 1), (2250, 7)]
                .into_iter()
                .chain(std::iter::empty())
                .collect::<Vec<_>>()
        );
        assert_eq!(l.layers(), 2);
        l.toggle(3000); // back to jamming
        assert_eq!(l.state(), State::Playing);
        l.undo();
        assert_eq!(l.layers(), 1);
        assert_eq!(play(&mut l, 3000, 4000), vec![(3000, 1)]);
    }

    #[test]
    fn tap_tempo_follows_the_recent_taps() {
        // Speeding up from 90 to 140 bpm: the estimate must end near
        // the tempo of the last taps, not the average of all of them.
        let mut l: Looper<u8> = Looper::new();
        l.tap = true;
        l.toggle(0);
        let mut t = 1000u64;
        for k in 0..12 {
            let bpm = 90.0 + 50.0 * k as f32 / 11.0;
            l.record(t, 5);
            t += (60.0 * SR / bpm) as u64;
        }
        // The last *gap* was at ~135 bpm (the final tap has no gap
        // after it); the mean of all gaps is ~113.
        let live = l.tap_bpm().unwrap();
        assert!((live - 135.0).abs() < 6.0, "live estimate {live}");
        let last_tap = t - (60.0 * SR / 140.0) as u64;
        assert!(l.toggle(t));
        assert!((l.bpm - live).abs() < 0.01);
        // The next click lands one beat after the last tap.
        let beat = (60.0 * SR / l.bpm).round() as u64;
        let got = play(&mut l, t, last_tap + beat + 1);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].0, last_tap + beat);
    }

    #[test]
    fn tap_tempo_sets_the_bpm_and_keeps_tapping() {
        let mut l: Looper<u8> = Looper::new();
        l.tap = true;
        l.toggle(0);
        // Taps at 120 bpm (22050 samples) with one late one.
        for t in [0u64, 22050, 44100 + 900, 66150, 88200] {
            l.record(1000 + t, 5);
        }
        assert!(l.toggle(1000 + 100_000));
        assert!((l.bpm - 120.0).abs() < 2.0, "bpm {}", l.bpm);
        assert!(l.quantize && !l.tap);
        assert!((l.len_secs() - 2.0).abs() < 0.05, "len {}", l.len_secs());
        // It keeps tapping, exactly on the grid (the first event after
        // an unaligned start may fire late; every gap after is exact).
        let got = play(&mut l, 101_000, 101_000 + 2 * 88_200);
        let times: Vec<u64> = got.iter().map(|(t, _)| *t).collect();
        assert!(times.len() >= 8, "{times:?}");
        let gaps: Vec<u64> = times.windows(2).skip(1).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|g| *g == gaps[0]), "uneven: {times:?}");
        assert!(
            (gaps[0] as f32 / 22050.0 - 1.0).abs() < 0.02,
            "gap {}",
            gaps[0]
        );
    }

    #[test]
    fn quantize_snaps_hits_and_length() {
        let mut l: Looper<u8> = Looper::new();
        l.bpm = 120.0;
        l.quantize = true;
        let g = ((SR * 60.0 / 120.0) / 4.0).round() as u64; // a sixteenth
        let bar = 16 * g;
        l.toggle(0);
        // The first hit is beat one wherever it lands after the press.
        let origin = 700;
        l.record(origin, 1);
        l.record(origin + g + 30, 2); // a little late for the sixteenth
        let close = origin + bar + 300; // a little long for the bar
        l.toggle(close);
        assert_eq!(l.len_secs(), bar as f32 / SR);
        let got = play(&mut l, close, origin + 2 * bar + 1);
        // Beat one had already passed when the loop closed, so it is
        // skipped rather than sounded late; the second hit is snapped
        // to the sixteenth; the next pass starts on time.
        assert_eq!(got, vec![(origin + bar + g, 2), (origin + 2 * bar, 1)]);
    }
}
