"""Biblioteca de licks e clichês de guitarra por estilo.

Cada lick é uma frase curta (1 a 4 tempos) com:
  - alturas relativas à TÔNICA (em semitons) ou ao ACORDE corrente (índice da nota do acorde,
    contando para cima: 0 = fundamental, 1 = 3ª, 2 = 5ª, 3 = 7ª ou oitava...);
  - nota dupla opcional (segunda nota tocada junto): distância exata em semitons
    (ex.: +5 = 4ª acima, -12 = oitava abaixo) ou diatônica ("+3d", "-3d", "+6d", "-6d":
    3ª/6ª maior ou menor, a que pertencer à escala);
  - articulação: bend de tom/semitom, sem bend, ou "let ring" (deixa soar);
  - afinidade por gênero: chance relativa de aparecer em cada estilo. Gêneros não listados
    recebem OTHER_AFFINITY, então qualquer lick pode eventualmente cair em outro estilo.
"""

from __future__ import annotations

from dataclasses import dataclass

from .theory import Chord

# Articulações
AUTO, BEND2, BEND1, STRAIGHT, RING = 0, 1, 2, 3, 4
ART_NAMES = {AUTO: "automática", BEND2: "bend de tom", BEND1: "bend de semitom",
             STRAIGHT: "sem bend", RING: "deixa soar"}

OTHER_AFFINITY = 0.05


@dataclass(frozen=True)
class LickNote:
    start: float                 # em tempos, a partir do início do lick
    dur: float                   # em tempos
    pitch: int                   # semitons da tônica, ou índice de nota do acorde (ref="chord")
    dyad: int | str | None = None
    art: int = AUTO


@dataclass(frozen=True)
class Lick:
    key: str
    name: str
    genre: str                   # estilo de origem
    description: str
    notes: tuple[LickNote, ...]
    affinity: dict[str, float]   # gênero -> chance relativa (o de origem vale 1.0)
    ref: str = "tonic"           # "tonic" ou "chord"

    @property
    def beats(self) -> float:
        return max(n.start + n.dur for n in self.notes)

    @property
    def triplet(self) -> bool:
        """Ritmo em tercinas (precisa da grade 3, 6, 12...)."""
        return any(abs(n.start * 3 - round(n.start * 3)) < 1e-6 and
                   abs(n.start * 4 - round(n.start * 4)) > 1e-6 for n in self.notes)

    @property
    def min_dur(self) -> float:
        return min(n.dur for n in self.notes)

    @property
    def has_dyads(self) -> bool:
        return any(n.dyad is not None for n in self.notes)

    def weight(self, genre: str) -> float:
        return self.affinity.get(genre, OTHER_AFFINITY)


def N(start, dur, pitch, dyad=None, art=AUTO) -> LickNote:
    return LickNote(start, dur, pitch, dyad, art)


def _seq(pitches, step, start=0.0, art=AUTO, dyad=None):
    """Sequência de notas de mesma duração."""
    return [N(start + i * step, step, p, dyad, art) for i, p in enumerate(pitches)]


def _lick(key, name, genre, desc, notes, affinity, ref="tonic"):
    aff = {genre: 1.0} | affinity
    return Lick(key, name, genre, desc, tuple(notes), aff, ref)


T = 1 / 3      # tercina de colcheia
S = 1 / 6      # sextina

LICKS: list[Lick] = [
    # ------------------------------------------------------------------ Blues
    _lick("blues_bend_resolve", "Bend na 4ª resolvendo na tônica", "Blues",
          "Sobe a 4ª até a 5ª com bend, desce pela b3 e repousa na tônica com vibrato.",
          [N(0, 1, 7, art=BEND2), N(1, .5, 5), N(1.5, .5, 3), N(2, 2, 0)],
          {"Rock": .6, "Hard rock": .6, "Country": .3}),
    _lick("blues_triplet_descent", "Descida em tercinas pela escala blues", "Blues",
          "Tercinas descendo da oitava, passando pela blue note (b5).",
          _seq([12, 10, 7, 6, 5, 3], T) + [N(2, 2, 0)],
          {"Rock": .5, "Hard rock": .5, "Metal": .2}),
    _lick("blues_b3_to_3", "Bend da b3 para a 3ª", "Blues",
          "Meio tom de bend da terça menor para a maior, alternando com a tônica.",
          [N(0, .5, 4, art=BEND1), N(.5, .5, 0), N(1, .5, 4, art=BEND1), N(1.5, .5, 0),
           N(2, 1, -2), N(3, 1, 0)],
          {"Country": .5, "Rock": .4}),
    _lick("blues_turnaround", "Turnaround com notas duplas", "Blues",
          "Tônica presa no agudo enquanto a voz de baixo desce cromaticamente.",
          [N(0, .5, 12, -2), N(.5, .5, 12, -3), N(1, .5, 12, -4), N(1.5, .5, 12, -5),
           N(2, 2, 7)],
          {"Rock": .3, "Country": .3}),
    _lick("blues_bb_box", "Box do B.B. King", "Blues",
          "Bend da 9ª para a 3ª na região da oitava, fechando na tônica.",
          [N(0, 1, 16, art=BEND2), N(1, .5, 14), N(1.5, .5, 12), N(2, .5, 14),
           N(2.5, 1.5, 12)],
          {"Pop": .3, "Rock": .3}),

    # ------------------------------------------------------------------ Rock
    _lick("rock_berry", "Notas duplas estilo Chuck Berry", "Rock",
          "Tônica com a 5ª abaixo repetida, bend na 4ª e retorno às notas duplas.",
          _seq([12] * 4, .5, dyad=-5) + [N(2, 1, 7, art=BEND2), N(3, 1, 12, -5)],
          {"Blues": .5, "Punk": .5, "Country": .4, "Hard rock": .5}),
    _lick("rock_page_sextuplets", "Repetição em sextinas (estilo Page)", "Rock",
          "Célula de três notas da pentatônica repetida rápido, terminando em bend.",
          _seq([12, 10, 7] * 4, S) + [N(2, 2, 12, art=BEND2)],
          {"Hard rock": 1.0, "Blues": .4, "Metal": .3}),
    _lick("rock_bend_4th", "Bend com 4ª presa", "Rock",
          "Bend de tom na 5ª enquanto a tônica acima fica presa (a 4ª entre as duas).",
          [N(0, 1.5, 7, 5, BEND2), N(1.5, .5, 7, 5), N(2, .5, 5), N(2.5, .5, 3), N(3, 1, 0)],
          {"Hard rock": .8, "Country": .5, "Blues": .4}),

    # ------------------------------------------------------------------ Hard rock
    _lick("hr_pedal_point", "Nota pedal alternada", "Hard rock",
          "Semicolcheias alternando a tônica aguda com uma linha descendente.",
          _seq([12, 7, 12, 5, 12, 3, 12, 2], .25) + [N(2, 2, 0)],
          {"Metal": .8}),
    _lick("hr_trill", "Trinado de 3 notas (hammer-on e pull-off)", "Hard rock",
          "Tônica, 2ª e 3ª rolando em semicolcheias (grupos de 3 contra o pulso de 4), "
          "resolvendo na tônica.",
          _seq([12, 14, 16] * 2 + [12, 14], .25, art=STRAIGHT) + [N(2, 2, 12)],
          {"Metal": .6, "Rock": .5}),

    # ------------------------------------------------------------------ Metal
    _lick("metal_sweep", "Arpejo varrido (sweep)", "Metal",
          "Notas do acorde subindo e descendo em semicolcheias por três oitavas.",
          _seq([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], .25, art=STRAIGHT) + [N(3, 1, 0)],
          {"Hard rock": .4, "Pop": .1}, ref="chord"),
    _lick("metal_tremolo", "Tremolo picking", "Metal",
          "Palhetada alternada contínua na tônica e na 2ª menor (b2 frígio).",
          _seq([12] * 4 + [13] * 4 + [15] * 4 + [13] * 4, .25, art=STRAIGHT),
          {"Punk": .3}),
    _lick("metal_neoclassic", "Corrida neoclássica em grupos de 4", "Metal",
          "Escala menor harmônica descendo em sequência de quatro notas.",
          _seq([19, 17, 15, 14, 17, 15, 14, 12, 15, 14, 12, 11], .25, art=STRAIGHT)
          + [N(3, 1, 12)],
          {"Hard rock": .3}),
    _lick("metal_twin_gallop", "Galope com harmonia em 3ªs", "Metal",
          "Ritmo de galope com a segunda guitarra uma 3ª acima (estilo twin guitar).",
          [N(0, .5, 7, "+3d"), N(.5, .25, 5, "+3d"), N(.75, .25, 7, "+3d"),
           N(1, .5, 3, "+3d"), N(1.5, .25, 2, "+3d"), N(1.75, .25, 3, "+3d"),
           N(2, 2, 0, "+3d")],
          {"Hard rock": .5}),

    # ------------------------------------------------------------------ Punk
    _lick("punk_octaves", "Melodia em oitavas", "Punk",
          "Melodia simples dobrada uma oitava abaixo.",
          [N(0, 1, 12, -12), N(1, .5, 10, -12), N(1.5, .5, 7, -12), N(2, 1, 5, -12),
           N(3, 1, 7, -12)],
          {"Indie": .6, "Reggae": .6, "Rock": .4}),

    # ------------------------------------------------------------------ Indie
    _lick("indie_ring_arp", "Arpejo com cordas soando", "Indie",
          "Notas do acorde em colcheias, deixando cada uma soar sobre a próxima.",
          _seq([0, 2, 4, 3, 1, 2], .5, art=RING),
          {"Pop": .6, "Rock": .2}, ref="chord"),
    _lick("indie_drone", "Nota pedal aguda (drone)", "Indie",
          "Linha descendo por baixo de uma tônica aguda que fica soando.",
          [N(0, 1, 7, 5), N(1, 1, 5, 7), N(2, 1, 4, 8), N(3, 1, 2, 10)],
          {"Pop": .5, "Rock": .3}),

    # ------------------------------------------------------------------ Reggae
    _lick("reggae_offbeat", "Frase nos contratempos", "Reggae",
          "Notas só nos contratempos, deixando os tempos vazios.",
          [N(.5, .5, 12), N(1.5, .5, 10), N(2.5, .5, 7), N(3.5, .5, 10)],
          {"Funk": .3}),

    # ------------------------------------------------------------------ Country
    _lick("country_pedal_steel", "Bend estilo pedal steel", "Country",
          "Bend de tom até a 3ª com a 5ª presa acima, imitando a pedal steel.",
          [N(0, 1.5, 4, 3, BEND2), N(1.5, .5, 4, 3), N(2, 1, 2, 5), N(3, 1, 0, 4)],
          {"Rock": .3, "Pop": .2}),
    _lick("country_chicken", "Chicken picking", "Country",
          "Notas curtas e estaladas com pausas, descendo para a tônica.",
          [N(0, .25, 12, art=STRAIGHT), N(.5, .25, 9, art=STRAIGHT),
           N(.75, .25, 7, art=STRAIGHT), N(1, .25, 12, art=STRAIGHT),
           N(1.5, .25, 9, art=STRAIGHT), N(1.75, .25, 7, art=STRAIGHT),
           N(2, .5, 4), N(2.5, .5, 2), N(3, 1, 0)],
          {"Funk": .3, "Rock": .2}),
    _lick("country_run", "Corrida maior com b3 de passagem", "Country",
          "Pentatônica maior em semicolcheias com a b3 cromática no caminho.",
          _seq([0, 2, 3, 4, 7, 9, 7, 4], .25) + [N(2, 2, 0)],
          {"Blues": .3, "Rock": .3}),

    # ------------------------------------------------------------------ Funk
    _lick("funk_dorian", "Frase sincopada dórica", "Funk",
          "Semicolcheias sincopadas destacando a 6ª maior do dórico.",
          [N(0, .25, 0), N(.5, .25, 3), N(.75, .25, 5), N(1.25, .25, 7), N(1.5, .5, 9),
           N(2.5, .25, 10), N(2.75, .25, 12), N(3, 1, 10)],
          {"Reggae": .3, "Indie": .2}),
    _lick("funk_sixths", "Notas duplas em 6ªs (soul)", "Funk",
          "Melodia com a 6ª diatônica abaixo, como nas guitarras de soul.",
          [N(0, .5, 12, "-6d"), N(.5, .5, 10, "-6d"), N(1, 1, 9, "-6d"),
           N(2, .5, 7, "-6d"), N(2.5, 1.5, 4, "-6d")],
          {"Pop": .6, "Country": .4, "Blues": .3}),

    # ------------------------------------------------------------------ Pop / balada
    _lick("pop_slow_bend", "Bend longo e melódico", "Pop",
          "Frase cantada com um bend de tom sustentado no meio.",
          [N(0, .5, 4), N(.5, .5, 7), N(1, 2, 9, art=BEND2), N(3, 1, 7)],
          {"Hard rock": .4, "Rock": .3}),
    _lick("pop_thirds", "Melodia em 3ªs", "Pop",
          "Graus conjuntos com a 3ª diatônica acima.",
          [N(0, 1, 0, "+3d"), N(1, 1, 2, "+3d"), N(2, 1, 4, "+3d"), N(3, 1, 2, "+3d")],
          {"Country": .5, "Indie": .4}),
]
LICKS_BY_KEY = {lk.key: lk for lk in LICKS}


# ------------------------------------------------------------------ realização

@dataclass(frozen=True)
class SoloEvent:
    """Nota já concreta do solo: o formato comum para licks avulsos e para o genoma."""
    start: float          # tempos (grade reta; o swing é aplicado na renderização)
    dur: float
    pitch: int            # MIDI
    dyad: int | None = None   # altura MIDI da nota dupla
    art: int = AUTO


def chord_tone_ladder(chord: Chord, octaves: int = 3) -> list[int]:
    """Intervalos das notas do acorde subindo por várias oitavas (para ref="chord")."""
    base = sorted(set(chord.intervals))
    return [i + 12 * o for o in range(octaves) for i in base]


def resolve_dyad(pitch: int, spec: int | str | None, scale_pcs: set[int]) -> int | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return pitch + spec
    sign = 1 if spec[0] == "+" else -1
    cands = {"3d": (3, 4), "6d": (8, 9)}[spec[1:]]
    for c in cands:
        if (pitch + sign * c) % 12 in scale_pcs:
            return pitch + sign * c
    return pitch + sign * cands[1]


def lick_offsets(lick: Lick, chord: Chord | None = None) -> list[int]:
    """Semitons de cada nota em relação à referência (tônica ou fundamental do acorde)."""
    if lick.ref == "chord":
        ladder = chord_tone_ladder(chord)
        return [ladder[n.pitch % len(ladder)] for n in lick.notes]
    return [n.pitch for n in lick.notes]


def realize_lick(lick: Lick, ref_pc: int, scale_pcs: set[int], chord: Chord | None = None,
                 low: int = 52, high: int = 88, near: int = 67,
                 start: float = 0.0) -> list[SoloEvent]:
    """Lick -> eventos com alturas MIDI, escolhendo a oitava que cabe em [low, high]
    e fica mais perto de `near`. As alturas saem exatamente como escritas (sem ajuste à
    escala); o encaixe na escala do solo é feito pelo GA ao inserir o lick no genoma."""
    offs = lick_offsets(lick, chord)
    bases = [b for b in range(low - 24, high + 1) if b % 12 == ref_pc
             and b + min(offs) >= low and b + max(offs) <= high]
    if not bases:
        bases = [min((b for b in range(low - 24, high + 1) if b % 12 == ref_pc),
                     key=lambda b: abs(b + (min(offs) + max(offs)) / 2 - near))]
    base = min(bases, key=lambda b: abs(b + sum(offs) / len(offs) - near))
    out = []
    for n, off in zip(lick.notes, offs):
        p = base + off
        out.append(SoloEvent(start + n.start, n.dur, p, resolve_dyad(p, n.dyad, scale_pcs),
                             n.art))
    return out
