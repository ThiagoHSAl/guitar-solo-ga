"""Presets rítmicos: padrão de acompanhamento, linha de baixo, bateria e feel (swing).

Todos os tempos são em batidas (semínimas) dentro de um compasso 4/4, numa grade *reta*.
O swing é aplicado só na conversão para MIDI (ver `swing_time`), então o mesmo padrão
pode ser ouvido reto ou com shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bateria General MIDI (canal 10)
KICK, SNARE, SIDESTICK, HAT_CLOSED, HAT_PEDAL, HAT_OPEN = 36, 38, 37, 42, 44, 46
CRASH, RIDE, RIDE_BELL, TOM_LOW, TOM_MID, TOM_HIGH = 49, 51, 53, 45, 47, 50

# Voicing de um ataque do acompanhamento:
#   "chord"        -> acorde completo (voicing fechado com condução de vozes)
#   "arpN"         -> N-ésima nota do voicing estendido (arpejo)
#   (i, j, ...)    -> "power chord": intervalos a partir da fundamental no registro grave
Voicing = str | tuple[int, ...]


@dataclass(frozen=True)
class CompHit:
    start: float
    dur: float
    vel: int
    voicing: Voicing = "chord"


@dataclass(frozen=True)
class BassHit:
    start: float
    dur: float
    vel: int
    degree: str = "1"   # "1", "3", "5", "6", "b7", "8" (3ª e 7ª seguem a qualidade do acorde)


@dataclass(frozen=True)
class DrumHit:
    start: float
    note: int
    vel: int


@dataclass(frozen=True)
class RhythmPreset:
    key: str
    label: str
    description: str
    swing: float                 # posição da colcheia "e": 0.5 = reto, 2/3 = shuffle de tercina
    comp_program: int            # programa GM do acompanhamento
    bass_program: int
    comp: list[CompHit]
    bass: list[BassHit]
    drums: list[DrumHit]
    default_bpm: int
    comp_register: int = 52      # nota MIDI em torno da qual o voicing é centrado
    genre: str = ""
    solo_subdiv: int = 2         # grade do solo: 2 = colcheias, 3 = tercinas, 4 = semicolcheias


def swing_time(beat_pos: float, swing: float) -> float:
    """Desloca a colcheia do contratempo de 0.5 para `swing` (e escala as subdivisões)."""
    if swing == 0.5:
        return beat_pos
    beat = int(beat_pos // 1)
    f = beat_pos - beat
    if f < 0.5:
        f = f * (swing / 0.5)
    else:
        f = swing + (f - 0.5) * ((1 - swing) / 0.5)
    return beat + f


def _eighths(fn):
    return [fn(i, i * 0.5) for i in range(8)]


def _shuffle() -> RhythmPreset:
    # Boogie clássico de guitarra: 1-5, 1-5, 1-6, 1-6, 1-b7, 1-b7, 1-6, 1-6 em colcheias swingadas.
    tops = [7, 7, 9, 9, 10, 10, 9, 9]
    comp = _eighths(lambda i, t: CompHit(t, 0.4, 92 if i % 2 == 0 else 74, (0, tops[i])))
    bass = [BassHit(0, 0.9, 100, "1"), BassHit(1, 0.9, 88, "3"),
            BassHit(2, 0.9, 96, "5"), BassHit(3, 0.9, 88, "6")]
    drums = [DrumHit(0, KICK, 105), DrumHit(2, KICK, 98),
             DrumHit(1, SNARE, 100), DrumHit(3, SNARE, 104)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_CLOSED, 88 if i % 2 == 0 else 62))
    return RhythmPreset("shuffle", "Blues shuffle", "Boogie de 5ª/6ª com swing de tercina",
                        swing=2 / 3, comp_program=27, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=100,
                        comp_register=45, genre="Blues")


def _slow_blues() -> RhythmPreset:
    # Feel 12/8: tercinas em cada tempo, escritas diretamente (swing desligado).
    trip = [b + k / 3 for b in range(4) for k in range(3)]
    comp = [CompHit(t, 0.3, 80 if abs(t - round(t)) < 1e-6 else 52, "chord") for t in trip]
    bass = [BassHit(0, 0.9, 100, "1"), BassHit(1, 0.6, 85, "3"), BassHit(5 / 3, 0.3, 75, "5"),
            BassHit(2, 0.9, 95, "5"), BassHit(3, 0.6, 85, "6"), BassHit(11 / 3, 0.3, 75, "5")]
    drums = [DrumHit(0, KICK, 100), DrumHit(2, KICK, 92), DrumHit(5 / 3, KICK, 70),
             DrumHit(1, SNARE, 96), DrumHit(3, SNARE, 100)]
    drums += [DrumHit(t, RIDE, 78 if abs(t - round(t)) < 1e-6 else 55) for t in trip]
    return RhythmPreset("slow_blues", "Slow blues 12/8", "Tercinas no piano elétrico e no ride",
                        swing=0.5, comp_program=4, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=62,
                        comp_register=58, genre="Blues", solo_subdiv=3)


def _rock() -> RhythmPreset:
    # Power chords em colcheias retas (palm mute), acento nos tempos.
    comp = _eighths(lambda i, t: CompHit(t, 0.35, 96 if i % 2 == 0 else 80, (0, 7, 12)))
    bass = _eighths(lambda i, t: BassHit(t, 0.4, 100 if i % 2 == 0 else 85, "1"))
    drums = [DrumHit(0, KICK, 110), DrumHit(2, KICK, 105), DrumHit(2.5, KICK, 92),
             DrumHit(1, SNARE, 110), DrumHit(3, SNARE, 112)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_CLOSED, 90 if i % 2 == 0 else 70))
    return RhythmPreset("rock", "Rock reto", "Power chords em colcheias, bumbo no 1, 3 e 3&",
                        swing=0.5, comp_program=29, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=120,
                        comp_register=45, genre="Rock")


def _funk() -> RhythmPreset:
    # Semicolcheias retas; ataques curtos e sincopados.
    comp = [CompHit(t, 0.15, v, "chord") for t, v in
            [(0, 100), (0.75, 78), (1.5, 88), (2.25, 80), (2.75, 92), (3.5, 85)]]
    bass = [BassHit(0, 0.4, 110, "1"), BassHit(0.75, 0.2, 90, "8"), BassHit(1.5, 0.2, 85, "b7"),
            BassHit(2, 0.4, 100, "1"), BassHit(2.75, 0.2, 90, "5"), BassHit(3.5, 0.4, 92, "b7")]
    drums = [DrumHit(0, KICK, 110), DrumHit(0.75, KICK, 88), DrumHit(2.5, KICK, 100),
             DrumHit(1, SNARE, 110), DrumHit(3, SNARE, 110),
             DrumHit(1.75, SNARE, 40), DrumHit(3.25, SNARE, 38)]
    drums += [DrumHit(i * 0.25, HAT_CLOSED, 85 if i % 2 == 0 else 50) for i in range(16)]
    return RhythmPreset("funk", "Funk", "Stabs sincopados e hi-hat em semicolcheias",
                        swing=0.5, comp_program=28, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=96,
                        comp_register=62, genre="Funk", solo_subdiv=4)


def _ballad() -> RhythmPreset:
    # Arpejo de piano em colcheias retas; baixo em mínimas.
    order = [0, 2, 4, 3, 5, 3, 4, 2]
    comp = _eighths(lambda i, t: CompHit(t, 0.9, 70 if i % 2 == 0 else 58, f"arp{order[i]}"))
    bass = [BassHit(0, 1.9, 92, "1"), BassHit(2, 1.9, 82, "5")]
    drums = [DrumHit(0, KICK, 90), DrumHit(2.5, KICK, 78),
             DrumHit(1, SIDESTICK, 80), DrumHit(3, SIDESTICK, 82)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_CLOSED, 60 if i % 2 == 0 else 42))
    return RhythmPreset("ballad", "Balada", "Arpejo de piano, baixo em mínimas, aro na caixa",
                        swing=0.5, comp_program=0, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=72,
                        comp_register=55, genre="Pop", solo_subdiv=4)


# ------------------------------------------------------------------ hard rock / metal

def _power(starts_durs_vels, shape=(0, 7, 12)):
    return [CompHit(t, d, v, shape) for t, d, v in starts_durs_vels]


def _hard_rock() -> RhythmPreset:
    # Riff de acordes abertos com respiro (estilo AC/DC): ataques longos no 1 e no 3,
    # empurrão no 2& e no 4&.
    comp = _power([(0, 1.3, 108), (1.5, 0.4, 92), (2, 1.3, 104), (3.5, 0.45, 96)])
    bass = _eighths(lambda i, t: BassHit(t, 0.42, 102 if i % 2 == 0 else 88, "1"))
    drums = [DrumHit(0, KICK, 110), DrumHit(2, KICK, 106), DrumHit(2.5, KICK, 94),
             DrumHit(1, SNARE, 112), DrumHit(3, SNARE, 114)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_OPEN if i == 7 else HAT_CLOSED,
                                           92 if i % 2 == 0 else 74))
    return RhythmPreset("hard_rock", "Hard rock", "Acordes abertos com respiro, hi-hat abrindo no 4&",
                        swing=0.5, comp_program=29, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=126,
                        comp_register=45, genre="Hard rock")


def _hard_rock_shuffle() -> RhythmPreset:
    # Shuffle pesado (estilo Led Zeppelin / ZZ Top): power chords swingados no tempo e no "e".
    comp = _eighths(lambda i, t: CompHit(t, 0.38 if i % 2 else 0.6, 104 if i % 2 == 0 else 84,
                                         (0, 7, 12)))
    bass = _eighths(lambda i, t: BassHit(t, 0.4, 100 if i % 2 == 0 else 84,
                                         "1" if i not in (6, 7) else ("b7" if i == 6 else "5")))
    drums = [DrumHit(0, KICK, 112), DrumHit(1.5, KICK, 90), DrumHit(2, KICK, 108),
             DrumHit(1, SNARE, 114), DrumHit(3, SNARE, 116)]
    drums += _eighths(lambda i, t: DrumHit(t, RIDE, 90 if i % 2 == 0 else 66))
    return RhythmPreset("hard_rock_shuffle", "Hard rock shuffle",
                        "Power chords swingados, ride no lugar do hi-hat",
                        swing=2 / 3, comp_program=29, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=112,
                        comp_register=45, genre="Hard rock")


def _gallop(t0: float) -> list[float]:
    return [t0, t0 + 0.5, t0 + 0.75]


def _heavy_gallop() -> RhythmPreset:
    # Heavy metal tradicional (NWOBHM): galope colcheia + 2 semicolcheias, palm mute.
    pos = [t for b in range(4) for t in _gallop(b)]
    comp = [CompHit(t, 0.22, 104 if t % 1 == 0 else 86, (0, 7)) for t in pos]
    bass = [BassHit(t, 0.22, 104 if t % 1 == 0 else 88, "1") for t in pos]
    drums = [DrumHit(t, KICK, 108 if t % 1 == 0 else 92) for t in pos]
    drums += [DrumHit(1, SNARE, 116), DrumHit(3, SNARE, 118)]
    drums += [DrumHit(b, RIDE_BELL, 92) for b in range(4)]
    return RhythmPreset("metal_gallop", "Heavy metal (galope)",
                        "Galope de colcheia + duas semicolcheias na guitarra, baixo e bumbo",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=150,
                        comp_register=45, genre="Metal")


def _thrash() -> RhythmPreset:
    # Thrash: palhetada alternada em semicolcheias com palm mute e "skank beat"
    # (bumbo nos tempos, caixa nos contratempos).
    comp = [CompHit(i * 0.25, 0.18, 104 if i % 4 == 0 else (90 if i % 2 == 0 else 80), (0, 7))
            for i in range(16)]
    bass = [BassHit(i * 0.25, 0.2, 100 if i % 2 == 0 else 86, "1") for i in range(16)]
    drums = [DrumHit(b, KICK, 110) for b in range(4)]
    drums += [DrumHit(b + 0.5, SNARE, 108) for b in range(4)]
    drums += [DrumHit(b, CRASH if b == 0 else RIDE, 96) for b in range(4)]
    return RhythmPreset("thrash", "Thrash metal",
                        "Palm mute em semicolcheias e skank beat (caixa no contratempo)",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=180,
                        comp_register=45, genre="Metal")


def _power_metal() -> RhythmPreset:
    # Power metal: bumbo duplo em semicolcheias contínuas, acordes abertos em colcheias.
    comp = _eighths(lambda i, t: CompHit(t, 0.45, 100 if i % 2 == 0 else 88, (0, 7, 12)))
    bass = _eighths(lambda i, t: BassHit(t, 0.45, 100 if i % 2 == 0 else 90, "1"))
    drums = [DrumHit(i * 0.25, KICK, 96 if i % 2 == 0 else 84) for i in range(16)]
    drums += [DrumHit(1, SNARE, 118), DrumHit(3, SNARE, 118)]
    drums += [DrumHit(b, RIDE, 94) for b in range(4)]
    return RhythmPreset("power_metal", "Power metal (bumbo duplo)",
                        "Bumbo duplo em semicolcheias, acordes abertos em colcheias",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=160,
                        comp_register=45, genre="Metal")


def _doom() -> RhythmPreset:
    # Doom: andamento lento, acordes sustentados e bateria em meio-tempo (caixa só no 3).
    comp = _power([(0, 1.9, 110), (2, 1.4, 100), (3.5, 0.48, 92)])
    bass = [BassHit(0, 1.9, 108, "1"), BassHit(2, 1.4, 100, "1"), BassHit(3.5, 0.48, 92, "1")]
    drums = [DrumHit(0, KICK, 112), DrumHit(1.5, KICK, 96), DrumHit(3.5, KICK, 90),
             DrumHit(2, SNARE, 118)]
    drums += [DrumHit(b, RIDE, 86) for b in range(4)]
    return RhythmPreset("doom", "Doom metal",
                        "Acordes sustentados, bateria em meio-tempo, andamento arrastado",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=64,
                        comp_register=43, genre="Metal", solo_subdiv=4)


# ------------------------------------------------------------------ punk / indie

def _punk() -> RhythmPreset:
    # Punk rock: power chords abertos em colcheias, só palhetada para baixo, tudo forte.
    comp = _eighths(lambda i, t: CompHit(t, 0.45, 104 if i % 2 == 0 else 98, (0, 7, 12)))
    bass = _eighths(lambda i, t: BassHit(t, 0.45, 102, "1"))
    drums = [DrumHit(0, KICK, 110), DrumHit(2, KICK, 108), DrumHit(2.5, KICK, 96),
             DrumHit(1, SNARE, 116), DrumHit(3, SNARE, 116)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_OPEN if i % 2 == 0 else HAT_CLOSED, 90))
    return RhythmPreset("punk", "Punk rock", "Colcheias abertas e rápidas, hi-hat meio aberto",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=180,
                        comp_register=45, genre="Punk")


def _skate_punk() -> RhythmPreset:
    # Skate/hardcore melódico: "polka beat" (bumbo nos tempos, caixa nos contratempos).
    comp = _eighths(lambda i, t: CompHit(t, 0.4, 102 if i % 2 == 0 else 92, (0, 7, 12)))
    bass = _eighths(lambda i, t: BassHit(t, 0.4, 100, "1" if i < 6 else "5"))
    drums = [DrumHit(b, KICK, 104) for b in range(4)]
    drums += [DrumHit(b + 0.5, SNARE, 106) for b in range(4)]
    drums += [DrumHit(b, CRASH if b == 0 else RIDE, 92) for b in range(4)]
    return RhythmPreset("skate_punk", "Skate punk (polka beat)",
                        "Bumbo nos tempos, caixa nos contratempos, bem rápido",
                        swing=0.5, comp_program=30, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=200,
                        comp_register=45, genre="Punk")


def _indie_dance() -> RhythmPreset:
    # Indie/dance-punk: bumbo nos quatro tempos, hi-hat aberto no contratempo,
    # guitarra limpa arpejada e baixo em oitavas.
    order = [0, 2, 1, 3, 2, 4, 3, 2]
    comp = _eighths(lambda i, t: CompHit(t, 0.45, 82 if i % 2 == 0 else 70, f"arp{order[i]}"))
    bass = _eighths(lambda i, t: BassHit(t, 0.4, 98 if i % 2 == 0 else 86,
                                         "1" if i % 2 == 0 else "8"))
    drums = [DrumHit(b, KICK, 108) for b in range(4)]
    drums += [DrumHit(1, SNARE, 104), DrumHit(3, SNARE, 106)]
    drums += [DrumHit(b + 0.5, HAT_OPEN, 80) for b in range(4)]
    drums += [DrumHit(b, HAT_PEDAL, 60) for b in range(4)]
    return RhythmPreset("indie_dance", "Indie dance-punk",
                        "Bumbo nos quatro tempos, hi-hat aberto no contratempo, guitarra arpejada",
                        swing=0.5, comp_program=27, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=124,
                        comp_register=60, genre="Indie")


def _indie_strum() -> RhythmPreset:
    # Indie rock "jangle": acordes limpos com levada sincopada (1, 2&, 3&, 4),
    # bateria com bumbo sincopado e tom-chão no lugar do hi-hat.
    comp = [CompHit(t, d, v, "chord") for t, d, v in
            [(0, 1.4, 88), (1.5, 0.9, 76), (2.5, 0.45, 78), (3, 0.9, 80)]]
    bass = [BassHit(0, 1.4, 96, "1"), BassHit(1.5, 0.9, 84, "1"), BassHit(2.5, 0.45, 84, "5"),
            BassHit(3, 0.9, 88, "1")]
    drums = [DrumHit(0, KICK, 100), DrumHit(1.5, KICK, 88), DrumHit(2.5, KICK, 92),
             DrumHit(1, SNARE, 100), DrumHit(3, SNARE, 102)]
    drums += _eighths(lambda i, t: DrumHit(t, TOM_LOW, 70 if i % 2 == 0 else 52))
    return RhythmPreset("indie_strum", "Indie rock (jangle)",
                        "Acordes limpos sincopados, tom-chão conduzindo a levada",
                        swing=0.5, comp_program=27, bass_program=34,
                        comp=comp, bass=bass, drums=drums, default_bpm=112,
                        comp_register=60, genre="Indie")


# ------------------------------------------------------------------ reggae / country

def _reggae() -> RhythmPreset:
    # One drop: bumbo + aro só no 3; skank (acorde curto) nos tempos 2 e 4;
    # baixo melódico que deixa o tempo 1 vazio.
    comp = [CompHit(1, 0.2, 92, "chord"), CompHit(3, 0.2, 92, "chord")]
    bass = [BassHit(0.5, 0.4, 96, "1"), BassHit(1, 0.4, 90, "1"), BassHit(1.5, 0.4, 88, "3"),
            BassHit(2.5, 0.4, 94, "5"), BassHit(3, 0.4, 86, "6"), BassHit(3.5, 0.4, 88, "5")]
    drums = [DrumHit(2, KICK, 104), DrumHit(2, SIDESTICK, 100)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_CLOSED, 76 if i % 2 == 0 else 58))
    return RhythmPreset("reggae", "Reggae (one drop)",
                        "Skank nos tempos 2 e 4, bumbo e aro só no 3, baixo melódico",
                        swing=0.6, comp_program=27, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=76,
                        comp_register=62, genre="Reggae", solo_subdiv=4)


def _ska() -> RhythmPreset:
    # Ska: guitarra em todos os contratempos, walking bass, bumbo no 1 e 3.
    comp = [CompHit(b + 0.5, 0.2, 92, "chord") for b in range(4)]
    bass = [BassHit(0, 0.9, 98, "1"), BassHit(1, 0.9, 88, "3"), BassHit(2, 0.9, 94, "5"),
            BassHit(3, 0.9, 88, "6")]
    drums = [DrumHit(0, KICK, 100), DrumHit(2, KICK, 98),
             DrumHit(1, SNARE, 104), DrumHit(3, SNARE, 106)]
    drums += [DrumHit(b + 0.5, HAT_OPEN, 78) for b in range(4)]
    drums += [DrumHit(b, HAT_CLOSED, 64) for b in range(4)]
    return RhythmPreset("ska", "Ska", "Guitarra em todos os contratempos, walking bass",
                        swing=0.55, comp_program=27, bass_program=33,
                        comp=comp, bass=bass, drums=drums, default_bpm=150,
                        comp_register=62, genre="Reggae")


def _country_boom_chick() -> RhythmPreset:
    # Boom-chick: baixo alternando fundamental e 5ª nos tempos 1 e 3; violão no 2 e 4.
    comp = [CompHit(1, 0.4, 84, "chord"), CompHit(3, 0.4, 86, "chord"),
            CompHit(3.5, 0.3, 66, "chord")]
    bass = [BassHit(0, 0.9, 100, "1"), BassHit(2, 0.9, 92, "5")]
    drums = [DrumHit(0, KICK, 96), DrumHit(2, KICK, 92),
             DrumHit(1, SNARE, 96), DrumHit(3, SNARE, 98)]
    drums += _eighths(lambda i, t: DrumHit(t, HAT_CLOSED, 70 if i % 2 == 0 else 50))
    return RhythmPreset("country", "Country (boom-chick)",
                        "Baixo alternando fundamental e 5ª, violão nos tempos 2 e 4",
                        swing=0.5, comp_program=25, bass_program=32,
                        comp=comp, bass=bass, drums=drums, default_bpm=116,
                        comp_register=55, genre="Country")


def _country_train() -> RhythmPreset:
    # Train beat: caixa em semicolcheias com acento nos tempos 2 e 4 (estilo Johnny Cash).
    comp = [CompHit(1, 0.4, 86, "chord"), CompHit(3, 0.4, 88, "chord")]
    bass = [BassHit(0, 0.9, 100, "1"), BassHit(1, 0.9, 84, "5"),
            BassHit(2, 0.9, 96, "8"), BassHit(3, 0.9, 84, "5")]
    drums = [DrumHit(0, KICK, 96), DrumHit(2, KICK, 92)]
    drums += [DrumHit(i * 0.25, SNARE, 100 if i in (4, 12) else (62 if i % 2 == 0 else 40))
              for i in range(16)]
    return RhythmPreset("country_train", "Country (train beat)",
                        "Caixa em semicolcheias acentuando 2 e 4, baixo em semínimas",
                        swing=0.5, comp_program=25, bass_program=32,
                        comp=comp, bass=bass, drums=drums, default_bpm=108,
                        comp_register=55, genre="Country")


PRESETS: dict[str, RhythmPreset] = {p.key: p for p in (
    _shuffle(), _slow_blues(),
    _rock(), _hard_rock(), _hard_rock_shuffle(),
    _heavy_gallop(), _thrash(), _power_metal(), _doom(),
    _punk(), _skate_punk(),
    _indie_dance(), _indie_strum(),
    _reggae(), _ska(),
    _country_boom_chick(), _country_train(),
    _funk(), _ballad(),
)}

# Escala padrão do solo por ritmo, conforme o modo harmônico da progressão:
#   ritmo: (progressão maior, progressão menor, progressão de blues)
# Escolhas guiadas pela sonoridade típica de cada vertente.
SOLO_SCALES: dict[str, tuple[str, str, str]] = {
    "shuffle":           ("mixolydian", "dorian", "blues_mixed"),
    "slow_blues":        ("mixolydian", "aeolian", "blues_mixed"),
    "rock":              ("mixolydian", "aeolian", "blues"),
    "hard_rock":         ("mixolydian", "dorian", "blues"),
    "hard_rock_shuffle": ("mixolydian", "dorian", "blues_mixed"),
    "metal_gallop":      ("ionian", "aeolian", "blues"),
    "thrash":            ("lydian", "phrygian", "blues"),
    "power_metal":       ("ionian", "harmonic_minor", "blues"),
    "doom":              ("mixolydian", "locrian", "blues"),
    "punk":              ("ionian", "aeolian", "blues"),
    "skate_punk":        ("major_pentatonic", "minor_pentatonic", "blues"),
    "indie_dance":       ("lydian", "aeolian", "mixolydian"),
    "indie_strum":       ("ionian", "dorian", "mixolydian"),
    "reggae":            ("mixolydian", "dorian", "mixolydian"),
    "ska":               ("ionian", "dorian", "mixolydian"),
    "country":           ("country", "aeolian", "country"),
    "country_train":     ("mixolydian", "dorian", "country"),
    "funk":              ("mixolydian", "dorian", "mixolydian"),
    "ballad":            ("lydian", "aeolian", "mixolydian"),
}
_MODE_INDEX = {"major": 0, "minor": 1, "blues": 2}


def default_solo_scale(rhythm: str, mode: str) -> str:
    return SOLO_SCALES[rhythm][_MODE_INDEX[mode]]


# Perfil de velocidade do solo por ritmo: (densidade em notas por tempo, fração de notas
# em rajadas rápidas - "shred"). Thrash e power metal pedem rajadas; reggae quase nenhuma.
SOLO_SPEED: dict[str, tuple[float, float]] = {
    "shuffle": (1.6, 0.15), "slow_blues": (1.4, 0.25),
    "rock": (1.8, 0.2), "hard_rock": (2.0, 0.3), "hard_rock_shuffle": (1.8, 0.25),
    "metal_gallop": (2.4, 0.4), "thrash": (2.6, 0.5), "power_metal": (2.8, 0.55),
    "doom": (1.4, 0.25),
    "punk": (1.5, 0.1), "skate_punk": (2.0, 0.25),
    "indie_dance": (1.2, 0.05), "indie_strum": (1.1, 0.05),
    "reggae": (0.9, 0.0), "ska": (1.5, 0.05),
    "country": (2.2, 0.35), "country_train": (2.0, 0.3),
    "funk": (1.6, 0.1), "ballad": (1.2, 0.2),
}


# Fração alvo de ataques com nota dupla (double stops, oitavas, 3ªs/6ªs) por gênero.
SOLO_DYADS: dict[str, float] = {
    "Blues": 0.1, "Rock": 0.15, "Hard rock": 0.1, "Metal": 0.12, "Punk": 0.15,
    "Indie": 0.12, "Reggae": 0.15, "Country": 0.25, "Funk": 0.1, "Pop": 0.1,
}


GENRES: list[str] = list(dict.fromkeys(p.genre for p in PRESETS.values()))


def presets_by_genre(genre: str) -> list[str]:
    return [k for k, p in PRESETS.items() if p.genre == genre]


def get_preset(key: str) -> RhythmPreset:
    if key not in PRESETS:
        raise ValueError(f"Ritmo desconhecido: {key!r}. Opções: {', '.join(PRESETS)}")
    return PRESETS[key]
