"""Teoria musical básica: notas, acordes, escalas, campo harmônico e progressões prontas."""

from __future__ import annotations

import re
from dataclasses import dataclass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_PC = {n: i for i, n in enumerate(NOTE_NAMES)} | {n: i for i, n in enumerate(FLAT_NAMES)}

# Qualidade do acorde -> intervalos em semitons a partir da fundamental.
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "5": (0, 7),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "9": (0, 4, 7, 10, 14),
    "m9": (0, 3, 7, 10, 14),
}
# Grafias alternativas aceitas na entrada.
_ALIASES = {"M": "", "maj": "", "min": "m", "-": "m", "°": "dim", "o": "dim",
            "ø": "m7b5", "M7": "maj7", "-7": "m7", "min7": "m7", "°7": "dim7", "o7": "dim7"}

# Escalas disponíveis para o solo (intervalos a partir da tônica).
SCALES: dict[str, tuple[int, ...]] = {
    # Modos gregos (modos da escala maior)
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    # Menor harmônica e seu 5º modo (muito usados no metal neoclássico)
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    # Pentatônicas e escalas de blues
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "blues": (0, 3, 5, 6, 7, 10),
    "blues_mixed": (0, 3, 4, 5, 6, 7, 10),    # blues + 3ª maior (resolve na 3ª do I7)
    "country": (0, 2, 3, 4, 7, 9),            # pentatônica maior + b3 de passagem
}
SCALE_LABELS = {
    "ionian": "Jônio (maior)", "dorian": "Dórico", "phrygian": "Frígio", "lydian": "Lídio",
    "mixolydian": "Mixolídio", "aeolian": "Eólio (menor natural)", "locrian": "Lócrio",
    "harmonic_minor": "Menor harmônica", "phrygian_dominant": "Frígio dominante",
    "minor_pentatonic": "Pentatônica menor", "major_pentatonic": "Pentatônica maior",
    "blues": "Blues", "blues_mixed": "Blues com 3ª maior", "country": "Country (pent. maior + b3)",
}
# Notas características de cada escala (intervalo a partir da tônica): é o que diferencia
# o modo dos vizinhos. Ficam isentas da penalidade de "nota que colide com o acorde" e são
# recompensadas pelo critério de cor modal da fitness.
SCALE_COLOR: dict[str, tuple[int, ...]] = {
    "ionian": (11,), "dorian": (9,), "phrygian": (1,), "lydian": (6,), "mixolydian": (10,),
    "aeolian": (8,), "locrian": (1, 6), "harmonic_minor": (11, 8),
    "phrygian_dominant": (1, 4), "minor_pentatonic": (), "major_pentatonic": (),
    "blues": (6,), "blues_mixed": (6, 3), "country": (3,),
}
# Escalas de 7 notas: recebem também as notas dos acordes da progressão (ex.: a sensível
# do V7 num tom menor). Pentatônicas e blues ficam "puras" para manter a sonoridade.
HEPTATONIC = {k for k, v in SCALES.items() if len(v) == 7 and k != "blues_mixed"}


def scale_note_names(tonic: str, scale: str) -> list[str]:
    t = note_to_pc(tonic)
    flats = t in _FLAT_KEYS or "b" in tonic
    return [pc_name(t + i, flats) for i in SCALES[scale]]


@dataclass(frozen=True)
class Chord:
    root: int        # pitch class 0-11
    quality: str     # chave de CHORD_INTERVALS
    symbol: str      # grafia original (para exibição)

    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_INTERVALS[self.quality]

    @property
    def pitch_classes(self) -> list[int]:
        return [(self.root + i) % 12 for i in self.intervals]


_CHORD_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")


def note_to_pc(name: str) -> int:
    name = name[0].upper() + name[1:]
    if name not in _PC:
        raise ValueError(f"Nota desconhecida: {name}")
    return _PC[name]


def pc_name(pc: int, flats: bool = False) -> str:
    return (FLAT_NAMES if flats else NOTE_NAMES)[pc % 12]


def parse_chord(symbol: str) -> Chord:
    m = _CHORD_RE.match(symbol.strip())
    if not m:
        raise ValueError(f"Acorde inválido: {symbol!r}")
    letter, acc, qual = m.groups()
    qual = _ALIASES.get(qual, qual)
    if qual not in CHORD_INTERVALS:
        raise ValueError(f"Qualidade de acorde não suportada: {symbol!r}")
    return Chord(note_to_pc(letter + acc), qual, symbol.strip())


def chord_symbol(root: int, quality: str, flats: bool = False) -> str:
    return pc_name(root, flats) + quality


def parse_progression(text: str) -> list[list[Chord]]:
    """Um token por compasso; acordes separados por vírgula dividem o compasso igualmente.

    Ex.: "A7 D7 A7 A7 | D7 D7 A7 A7 | E7 D7 A7 E7"  ou  "Dm7 G7 Cmaj7 Am7,D7".
    Barras verticais são ignoradas (servem só para legibilidade).
    """
    bars = []
    for tok in text.replace("|", " ").split():
        bars.append([parse_chord(c) for c in tok.split(",") if c])
    if not bars:
        raise ValueError("Progressão vazia")
    return bars


def progression_to_text(bars: list[list[Chord]]) -> str:
    return " ".join(",".join(c.symbol for c in bar) for bar in bars)


# ---------------------------------------------------------------- campo harmônico

# (grau em semitons, qualidade tríade, qualidade tétrade, numeral romano)
_FIELDS = {
    "major": [(0, "", "maj7", "I"), (2, "m", "m7", "ii"), (4, "m", "m7", "iii"),
              (5, "", "maj7", "IV"), (7, "", "7", "V"), (9, "m", "m7", "vi"),
              (11, "dim", "m7b5", "vii°")],
    "minor": [(0, "m", "m7", "i"), (2, "dim", "m7b5", "ii°"), (3, "", "maj7", "III"),
              (5, "m", "m7", "iv"), (7, "m", "m7", "v"), (8, "", "maj7", "VI"),
              (10, "", "7", "VII"), (7, "", "7", "V (harm.)")],
    "blues": [(0, "7", "7", "I7"), (5, "7", "7", "IV7"), (7, "7", "7", "V7"),
              (2, "m7", "m7", "ii7"), (9, "7", "7", "VI7")],
}

MODES = list(_FIELDS)
# Tônicas em que é mais natural escrever com bemóis.
_FLAT_KEYS = {1, 3, 5, 8, 10}


def harmonic_field(tonic: str, mode: str, sevenths: bool = False) -> list[tuple[str, str]]:
    """Retorna [(numeral, símbolo do acorde)] do campo harmônico da tônica/modo."""
    t = note_to_pc(tonic)
    flats = t in _FLAT_KEYS or (mode == "minor" and t in {0, 2, 5, 7})
    out = []
    for deg, tri, tet, roman in _FIELDS[mode]:
        q = tet if sevenths else tri
        out.append((roman, chord_symbol((t + deg) % 12, q, flats)))
    return out


# Progressões prontas, em graus relativos à tônica: (semitons, qualidade) por compasso.
# Compassos com dois acordes são listas de duas tuplas.
PRESET_PROGRESSIONS: dict[str, dict] = {
    "12-bar blues": {"mode": "blues", "bars": [
        [(0, "7")], [(5, "7")], [(0, "7")], [(0, "7")],
        [(5, "7")], [(5, "7")], [(0, "7")], [(0, "7")],
        [(7, "7")], [(5, "7")], [(0, "7")], [(7, "7")]]},
    "12-bar blues menor": {"mode": "minor", "bars": [
        [(0, "m7")], [(0, "m7")], [(0, "m7")], [(0, "m7")],
        [(5, "m7")], [(5, "m7")], [(0, "m7")], [(0, "m7")],
        [(8, "7")], [(7, "7")], [(0, "m7")], [(7, "7")]]},
    "I-V-vi-IV (pop/rock)": {"mode": "major", "bars": [
        [(0, "")], [(7, "")], [(9, "m")], [(5, "")]] * 2},
    "i-VII-VI-VII (rock menor)": {"mode": "minor", "bars": [
        [(0, "m")], [(10, "")], [(8, "")], [(10, "")]] * 2},
    "ii-V-I (jazz)": {"mode": "major", "bars": [
        [(2, "m7")], [(7, "7")], [(0, "maj7")], [(0, "maj7")]] * 2},
    "i-iv-v (blues rock menor)": {"mode": "minor", "bars": [
        [(0, "m7")], [(0, "m7")], [(5, "m7")], [(5, "m7")],
        [(0, "m7")], [(0, "m7")], [(7, "m7")], [(5, "m7"), (7, "7")]]},
    "i-VI-VII (metal)": {"mode": "minor", "bars": [
        [(0, "5")], [(0, "5")], [(8, "5")], [(10, "5")]] * 2},
    "i-VII-VI-V (cadência andaluza)": {"mode": "minor", "bars": [
        [(0, "m")], [(10, "")], [(8, "")], [(7, "")]] * 2},
    "I-IV-V (punk / rock básico)": {"mode": "major", "bars": [
        [(0, "")], [(0, "")], [(5, "")], [(7, "")]] * 2},
    "I-IV-I-V (country)": {"mode": "major", "bars": [
        [(0, "")], [(0, "")], [(5, "")], [(5, "")],
        [(0, "")], [(0, "")], [(7, "")], [(7, "")]]},
    "i-iv (reggae menor)": {"mode": "minor", "bars": [
        [(0, "m")], [(5, "m")]] * 4},
    "I-vi-IV-V (indie / doo-wop)": {"mode": "major", "bars": [
        [(0, "")], [(9, "m")], [(5, "")], [(7, "")]] * 2},
}


def preset_progression(name: str, tonic: str) -> str:
    p = PRESET_PROGRESSIONS[name]
    t = note_to_pc(tonic)
    flats = t in _FLAT_KEYS or (p["mode"] == "minor" and t in {0, 2, 5, 7})
    return " ".join(
        ",".join(chord_symbol((t + d) % 12, q, flats) for d, q in bar) for bar in p["bars"]
    )
