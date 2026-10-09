"""Montagem e exportação dos arquivos MIDI (base, bateria, solo e mixagem)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from .backing import build_backing, build_drums
from .config import SongConfig
from .ga.genome import decode
from .licks import AUTO, BEND1, BEND2, RING, STRAIGHT, SoloEvent
from .render import SOLO_DB, balance_mix
from .rhythms import swing_time


def _new_midi(cfg: SongConfig, *instruments: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(cfg.bpm))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    pm.instruments.extend(instruments)
    return pm


def build_backing_midis(cfg: SongConfig, mix: bool = True) -> dict[str, pretty_midi.PrettyMIDI]:
    """MIDIs da etapa 1 (antes do solo): base, bateria e base+bateria."""
    comp, bass = build_backing(cfg)
    drums = build_drums(cfg)
    full = _new_midi(cfg, comp, bass, drums)
    if mix:
        balance_mix(full)
    return {
        "base": _new_midi(cfg, comp, bass),
        "bateria": _new_midi(cfg, drums),
        "base_bateria": full,
    }


def write_midis(midis: dict[str, pretty_midi.PrettyMIDI], out_dir: str | Path,
                prefix: str = "") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, pm in midis.items():
        path = out_dir / f"{prefix}{name}.mid"
        pm.write(str(path))
        paths[name] = path
    return paths


# ------------------------------------------------------------------ solo

# Timbre do solo por gênero (GM): 29 overdrive, 30 distorção, 27 guitarra limpa.
SOLO_PROGRAM = {"Metal": 30, "Hard rock": 30, "Punk": 30, "Rock": 29, "Blues": 29,
                "Indie": 27, "Reggae": 27, "Country": 27, "Funk": 27, "Pop": 29}
BEND_RANGE = 2.0          # semitons cobertos pelo pitch bend máximo (padrão GM)


def _bend_value(semitones: float) -> int:
    return int(max(-8192, min(8191, semitones / BEND_RANGE * 8192)))


def render_events(events: list[SoloEvent], cfg: SongConfig, swing: float, basic: float,
                  pitch_set: set[int] | None = None, sustain_after: float | None = None,
                  expressive: bool = True, seed: int = 0,
                  ) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument]:
    """Eventos do solo -> (instrumento principal, instrumento da 2ª voz).

    Os eventos vêm do genoma (via `genome_events`), já com nota dupla e articulação.
      - swing: aplicado aqui (os eventos estão numa grade reta);
      - basic: duração da figura "simples" em tempos (0,5 = colcheia, 1/3 = tercina), usada
        para acentuar o início de cada grupo dentro das rajadas;
      - sustain_after: notas que terminam neste ponto (em tempos) sustentam mais 3 tempos
        (a última nota do solo soa sobre o acorde final);
      - articulações: BEND2/BEND1 forçam bend de tom/semitom; STRAIGHT desliga bend e
        vibrato; RING deixa a nota soar sobre as seguintes; AUTO decide como antes
        (nota longa alcançada subindo ganha bend em metade dos casos; notas de um tempo ou
        mais ganham vibrato).
    A 2ª voz (nota dupla) fica em outra faixa: o bend da nota principal não a afeta.
    """
    rng = np.random.default_rng(seed)
    program = SOLO_PROGRAM.get(cfg.preset.genre, 29)
    main = pretty_midi.Instrument(program=program, name="Solo (GA)")
    second = pretty_midi.Instrument(program=program, name="Solo (2ª voz)")
    pitch_set = pitch_set or set()

    def sec(beat: float) -> float:
        bar, b = divmod(beat, 4)
        return (bar * 4 + swing_time(b, swing)) * cfg.beat_sec

    for k, ev in enumerate(events):
        start = sec(ev.start)
        end_beat = ev.start + ev.dur
        if sustain_after is not None and end_beat >= sustain_after - 1e-6:
            end = sec(sustain_after) + 3.0 * cfg.beat_sec
        elif ev.art == RING:
            end = sec(end_beat) + max(0.0, 1.5 - ev.dur) * cfg.beat_sec
        else:
            gap = min(0.02, 0.15 * (sec(end_beat) - start))   # notas rápidas quase em legato
            end = sec(end_beat) - gap
        on_beat = abs(ev.start - round(ev.start)) < 1e-6
        fast = ev.dur <= 0.26
        base_vel = 84 if fast else (100 if on_beat else 90)
        if fast and abs(ev.start / basic - round(ev.start / basic)) < 1e-6:
            base_vel += 8                                        # acento dentro da rajada
        vel = int(np.clip(base_vel + rng.integers(-5, 6), 1, 127))
        p = ev.pitch
        played, bend = p, 0.0
        dur_sec = end - start

        if expressive and ev.art != STRAIGHT:
            if ev.art == BEND2:
                bend = 2.0
            elif ev.art == BEND1:
                bend = 1.0
            elif ev.art == AUTO:
                rising = k > 0 and p > events[k - 1].pitch
                if rising and ev.dur >= 1 and (p - 2) in pitch_set and rng.random() < 0.5:
                    bend = 2.0
            if bend:
                played = p - int(bend)
                steps, t_bend = 8, min(0.15, dur_sec * 0.35)
                for i in range(steps + 1):
                    main.pitch_bends.append(pretty_midi.PitchBend(
                        _bend_value(bend * i / steps), start + t_bend * i / steps))
            if dur_sec >= cfg.beat_sec * 0.9 and ev.art != RING:
                t, rate, depth = start + min(0.25, dur_sec * 0.3), 5.5, 0.3
                while t < end:
                    v = bend + depth * np.sin(2 * np.pi * rate * (t - start))
                    main.pitch_bends.append(pretty_midi.PitchBend(_bend_value(v), t))
                    t += 0.02
            if bend or dur_sec >= cfg.beat_sec * 0.9:
                main.pitch_bends.append(pretty_midi.PitchBend(0, end))

        main.notes.append(pretty_midi.Note(vel, played, start, end))
        if ev.dyad is not None:
            second.notes.append(pretty_midi.Note(max(1, vel - 8), ev.dyad, start, end))
    main.pitch_bends.sort(key=lambda b: b.time)
    return main, second


def genome_events(genome, space) -> list[SoloEvent]:
    notes = decode(genome, space)
    sub = space.subdiv
    return [SoloEvent(o / sub, d / sub, int(p), int(p + dy) if dy else None, int(a))
            for o, d, p, dy, a in zip(notes.onset, notes.dur, notes.pitch, notes.dyad,
                                      notes.art)]


def build_solo(genome, space, cfg: SongConfig, expressive: bool = True,
               seed: int = 0) -> list[pretty_midi.Instrument]:
    """Genoma -> instrumentos MIDI do solo (principal e, se houver notas duplas, 2ª voz)."""
    events = genome_events(genome, space)
    main, second = render_events(
        events, cfg, space.swing, space.basic / space.subdiv, set(space.pitches.tolist()),
        sustain_after=space.n_slots / space.subdiv if cfg.ending else None,
        expressive=expressive, seed=seed)
    return [main] + ([second] if second.notes else [])


def build_song_midis(cfg: SongConfig, genome, space, expressive: bool = True,
                     seed: int = 0, mix: bool = True,
                     solo_db: float = SOLO_DB) -> dict[str, pretty_midi.PrettyMIDI]:
    """Todos os MIDIs: base, bateria, solo e a música completa."""
    comp, bass = build_backing(cfg)
    drums = build_drums(cfg)
    solo = build_solo(genome, space, cfg, expressive, seed)
    full = _new_midi(cfg, comp, bass, drums, *solo)
    if mix:
        balance_mix(full, solo_db=solo_db)   # CC7 nos instrumentos, compartilhados pelos 4 MIDIs
    return {
        "base": _new_midi(cfg, comp, bass),
        "bateria": _new_midi(cfg, drums),
        "solo": _new_midi(cfg, *solo),
        "completo": full,
    }
