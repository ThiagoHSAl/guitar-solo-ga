"""Geração determinística da base (acompanhamento + baixo) e da bateria a partir do SongConfig.

A base NÃO é produzida pelo algoritmo genético: é uma renderização da progressão de
entrada segundo o preset rítmico escolhido.
"""

from __future__ import annotations

import numpy as np
import pretty_midi

from .config import BEATS_PER_BAR, SongConfig
from .rhythms import (CRASH, HAT_OPEN, KICK, SNARE, TOM_HIGH, TOM_LOW, TOM_MID, DrumHit,
                      swing_time)
from .theory import Chord

BASS_LOW = 28            # E1: fundamental do baixo fica em [28, 39]
HUMANIZE_SEED = 1234     # base reproduzível: a "humanização" usa semente fixa


class _Timeline:
    """Converte (compasso, batida na grade reta) em segundos, aplicando swing e humanização."""

    def __init__(self, cfg: SongConfig, jitter_ms: float = 6.0):
        self.cfg = cfg
        self.rng = np.random.default_rng(HUMANIZE_SEED)
        self.jitter = jitter_ms / 1000.0

    def t(self, bar: int, beat: float, humanize: bool = True) -> float:
        beats = bar * BEATS_PER_BAR + swing_time(beat, self.cfg.swing)
        sec = beats * self.cfg.beat_sec
        if humanize and sec > 0:
            sec += self.rng.uniform(-self.jitter, self.jitter)
        return max(sec, 0.0)

    def dur(self, beats: float) -> float:
        return beats * self.cfg.beat_sec

    def vel(self, v: int, spread: int = 6) -> int:
        return int(np.clip(v + self.rng.integers(-spread, spread + 1), 1, 127))


# ------------------------------------------------------------------ voicings

def _closed_voicings(chord: Chord, center: int) -> list[list[int]]:
    """Todas as inversões em posição fechada com a nota mais grave perto de `center`."""
    pcs = chord.pitch_classes
    out = []
    for inv in range(len(pcs)):
        order = pcs[inv:] + pcs[:inv]
        for base in range(center - 8, center + 5):
            if base % 12 != order[0]:
                continue
            notes = [base]
            for pc in order[1:]:
                n = notes[-1] + 1
                while n % 12 != pc:
                    n += 1
                notes.append(n)
            out.append(notes)
    return out


def _voice_lead(chord: Chord, prev: list[int] | None, center: int) -> list[int]:
    cands = _closed_voicings(chord, center)

    def cost(v):
        c = abs(np.mean(v) - (center + 5)) * 0.5      # puxa para o registro do preset
        if prev:
            c += sum(min(abs(n - p) for p in prev) for n in v)
        return c

    return min(cands, key=cost)


def _bass_note(chord: Chord, degree: str) -> int:
    iv = chord.intervals
    if degree == "1":
        off = 0
    elif degree == "3":
        off = 4 if 4 in iv else 3      # sem terça (power chord): b3, mais segura no rock
    elif degree == "5":
        off = next((i for i in iv if i in (6, 7, 8)), 7)
    elif degree == "6":
        off = 9
    elif degree == "b7":
        off = 11 if 11 in iv else 10
    elif degree == "8":
        off = 12
    else:
        raise ValueError(degree)
    root = BASS_LOW + (chord.root - BASS_LOW) % 12
    return root + off


def _power_root(chord: Chord, register: int) -> int:
    low = register - 5
    return low + (chord.root - low) % 12


# ------------------------------------------------------------------ partes

def build_backing(cfg: SongConfig) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument]:
    """Retorna (acompanhamento, baixo)."""
    p = cfg.preset
    tl = _Timeline(cfg)
    comp = pretty_midi.Instrument(program=p.comp_program, name="Base - acompanhamento")
    bass = pretty_midi.Instrument(program=p.bass_program, name="Base - baixo")

    voicing_cache: dict[tuple[int, int], list[int]] = {}
    prev_voicing: list[int] | None = None

    def voicing_for(bar: int, chord: Chord) -> list[int]:
        nonlocal prev_voicing
        key = (bar, id(chord))
        if key not in voicing_cache:
            voicing_cache[key] = _voice_lead(chord, prev_voicing, p.comp_register)
            prev_voicing = voicing_cache[key]
        return voicing_cache[key]

    for bar in range(cfg.n_bars):
        for hit in p.comp:
            chord = cfg.chord_at(bar, hit.start)
            start = tl.t(bar, hit.start)
            end = start + tl.dur(hit.dur)
            if isinstance(hit.voicing, tuple):
                root = _power_root(chord, p.comp_register)
                notes = [root + i for i in hit.voicing]
            elif hit.voicing == "chord":
                notes = voicing_for(bar, chord)
            else:  # "arpN"
                v = voicing_for(bar, chord)
                ext = v + [n + 12 for n in v]
                notes = [ext[int(hit.voicing[3:]) % len(ext)]]
            for n in notes:
                comp.notes.append(pretty_midi.Note(tl.vel(hit.vel), n, start, end))

        for hit in p.bass:
            chord = cfg.chord_at(bar, hit.start)
            start = tl.t(bar, hit.start)
            bass.notes.append(pretty_midi.Note(tl.vel(hit.vel), _bass_note(chord, hit.degree),
                                               start, start + tl.dur(hit.dur)))

    if cfg.ending:
        _add_ending_chord(cfg, tl, comp, bass)
    return comp, bass


def _final_chord(cfg: SongConfig) -> Chord:
    # A progressão normalmente começa na tônica; o acorde final repete o primeiro.
    return cfg.bars[0][0]


def _add_ending_chord(cfg, tl, comp, bass):
    p = cfg.preset
    bar = cfg.n_bars
    chord = _final_chord(cfg)
    start, end = tl.t(bar, 0, humanize=False), tl.t(bar, 0, humanize=False) + tl.dur(3.5)
    if any(isinstance(h.voicing, tuple) for h in p.comp):
        root = _power_root(chord, p.comp_register)
        notes = [root, root + 7, root + 12]
    else:
        notes = _voice_lead(chord, None, p.comp_register)
    for n in notes:
        comp.notes.append(pretty_midi.Note(100, n, start, end))
    bass.notes.append(pretty_midi.Note(105, _bass_note(chord, "1"), start, end))


def _fill() -> list[DrumHit]:
    """Virada simples nos tempos 3 e 4: semicolcheias descendo pelos tons."""
    toms = [SNARE, SNARE, TOM_HIGH, TOM_HIGH, TOM_MID, TOM_MID, TOM_LOW, TOM_LOW]
    return [DrumHit(2 + i * 0.25, n, 80 + i * 4) for i, n in enumerate(toms)] + \
           [DrumHit(2, KICK, 95), DrumHit(3, KICK, 95)]


def build_drums(cfg: SongConfig, fills: bool = True) -> pretty_midi.Instrument:
    p = cfg.preset
    tl = _Timeline(cfg, jitter_ms=4.0)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Bateria")
    prog_len = len(cfg.bars)

    def add(bar, hit):
        start = tl.t(bar, hit.start)
        drums.notes.append(pretty_midi.Note(tl.vel(hit.vel, 5), hit.note, start, start + 0.1))

    for bar in range(cfg.n_bars):
        last_of_chorus = (bar % prog_len) == prog_len - 1
        do_fill = fills and last_of_chorus and prog_len >= 4
        for hit in p.drums:
            if do_fill and hit.start >= 2:
                continue
            add(bar, hit)
        if do_fill:
            for hit in _fill():
                add(bar, hit)
        if bar % prog_len == 0:
            add(bar, DrumHit(0, CRASH, 100))

    if cfg.ending:
        bar = cfg.n_bars
        for note in (CRASH, KICK, HAT_OPEN):
            add(bar, DrumHit(0, note, 112))
    return drums
