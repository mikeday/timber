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
    /// playing ↔ overdub. (Stopped resumes to playing.)
    pub fn toggle(&mut self, now: u64) {
        self.state = match self.state {
            State::Idle => {
                self.origin = now;
                self.layers.clear();
                self.pending.clear();
                self.merged.clear();
                State::Recording
            }
            State::Recording => {
                let mut len = (now - self.origin).max(1);
                if self.quantize {
                    // Whole bars, at least one.
                    let bar = 16 * self.sixteenth();
                    len = ((len + bar / 2) / bar).max(1) * bar;
                }
                self.len = len;
                self.commit();
                self.cursor = 0;
                self.last_pos = 0;
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
                let off = self.snap(now.saturating_sub(self.origin));
                self.pending.push((off, e));
            }
            State::Overdub => {
                let off = self.snap((now - self.origin) % self.len) % self.len;
                self.pending.push((off, e));
            }
            _ => {}
        }
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
        if t % beat == 0 {
            Some(t % (4 * beat) == 0)
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
        // Beat one had already passed when the loop closed (it sounds
        // late, once); the second hit is snapped to the sixteenth; the
        // next pass starts on time.
        assert_eq!(
            got,
            vec![(close, 1), (origin + bar + g, 2), (origin + 2 * bar, 1)]
        );
    }
}
