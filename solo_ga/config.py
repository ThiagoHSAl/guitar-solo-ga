"""Configuração única da música: alimenta base, bateria e (depois) a fitness do GA."""

from __future__ import annotations

from dataclasses import dataclass, field

from .rhythms import RhythmPreset, default_solo_scale, get_preset
from .theory import SCALES, Chord, parse_progression

BEATS_PER_BAR = 4


@dataclass
class SongConfig:
    progression: str                 # ex.: "A7 D7 A7 A7 D7 D7 A7 A7 E7 D7 A7 E7"
    rhythm: str = "shuffle"
    bpm: int | None = None           # None -> andamento padrão do preset
    tonic: str = "A"
    mode: str = "blues"
    choruses: int = 2                # quantas vezes a progressão se repete
    swing: float | None = None       # None -> feel do preset
    ending: bool = True              # compasso final com acorde da tônica + prato
    solo_scale: str = "auto"         # chave de SCALES, ou "auto" (padrão do ritmo + modo)
    bars: list[list[Chord]] = field(init=False, repr=False)

    def __post_init__(self):
        self.bars = parse_progression(self.progression)
        if self.bpm is None:
            self.bpm = self.preset.default_bpm
        if self.swing is None:
            self.swing = self.preset.swing
        if self.solo_scale != "auto" and self.solo_scale not in SCALES:
            raise ValueError(f"Escala desconhecida: {self.solo_scale!r}")

    @property
    def preset(self) -> RhythmPreset:
        return get_preset(self.rhythm)

    @property
    def scale(self) -> str:
        """Escala efetiva do solo."""
        if self.solo_scale == "auto":
            return default_solo_scale(self.rhythm, self.mode)
        return self.solo_scale

    @property
    def beat_sec(self) -> float:
        return 60.0 / self.bpm

    @property
    def n_bars(self) -> int:
        """Compassos do trecho com acompanhamento (sem o compasso final)."""
        return len(self.bars) * self.choruses

    def bar_chords(self, bar: int) -> list[Chord]:
        return self.bars[bar % len(self.bars)]

    def chord_at(self, bar: int, beat: float) -> Chord:
        """Acorde soando no tempo `beat` (0-4) do compasso `bar`."""
        chords = self.bar_chords(bar)
        idx = min(int(beat * len(chords) / BEATS_PER_BAR), len(chords) - 1)
        return chords[idx]
