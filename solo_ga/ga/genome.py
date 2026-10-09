"""Representação do genótipo do solo.

O solo é uma matriz de inteiros 4 x n_slots, com uma coluna por slot da grade rítmica
(n_compassos x 4 tempos x subdivisão). As linhas são:

    MEL   melodia:  REST (-1) pausa; HOLD (-2) sustenta a nota anterior (ou prolonga a
                    pausa); 0..P-1 ataque de nota, índice em `SoloSpace.pitches`
    DYAD  nota dupla tocada junto com o ataque: distância em semitons (0 = nenhuma)
    ART   articulação do ataque (ver solo_ga.licks: automática, bends, sem bend, deixa soar)
    SRC   origem: 1 + índice do lick que escreveu o slot (0 = não veio de lick). Não afeta
          o som nem a fitness; serve para relatar quais licks sobreviveram à evolução.

DYAD e ART só têm sentido nos slots com ataque de nota.

`SoloSpace` pré-calcula, por slot, tudo o que a fitness precisa saber sobre a base
(acorde corrente, notas do acorde, notas a evitar, ataques da base), para que a avaliação
seja só aritmética sobre arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BEATS_PER_BAR, SongConfig
from ..licks import LICKS, Lick, realize_lick, resolve_dyad
from ..rhythms import KICK, SNARE, swing_time
from ..theory import HEPTATONIC, SCALE_COLOR, SCALES, Chord, note_to_pc

REST = -1
HOLD = -2
MEL, DYAD, ART, SRC = 0, 1, 2, 3
N_ROWS = 4

# Extensão do solo (registro típico de guitarra solo): G3 .. D6
SOLO_LOW, SOLO_HIGH = 55, 86

# Notas de tensão aceitas no blues mesmo quando colidem com o acorde (b3 e b5).
_BLUE_NOTES = {3, 6}


def solo_scale(cfg: SongConfig) -> list[int]:
    """Classes de altura (0-11) permitidas no solo.

    Parte da escala efetiva (`cfg.scale`: escolhida pelo usuário ou padrão do ritmo/modo).
    Escalas de 7 notas recebem também as notas dos acordes da progressão, para que o solo
    sempre tenha as notas do acorde à disposição; pentatônicas e blues ficam puras.
    """
    t = note_to_pc(cfg.tonic)
    pcs = {(t + i) % 12 for i in SCALES[cfg.scale]}
    if cfg.scale in HEPTATONIC:
        for bar in cfg.bars:
            for ch in bar:
                pcs.update(ch.pitch_classes)
    return sorted(pcs)


@dataclass
class SoloSpace:
    cfg: SongConfig
    subdiv: int                 # slots por tempo
    n_slots: int
    pitches: np.ndarray         # (P,) alturas MIDI permitidas, crescentes
    chord_tone: np.ndarray      # (P, n_slots) bool: altura é nota do acorde corrente
    avoid: np.ndarray           # (P, n_slots) bool: altura colide com o acorde (b9 sobre nota do acorde)
    root_pc: np.ndarray         # (n_slots,) fundamental do acorde corrente
    accent: np.ndarray          # (n_slots,) bool: slot coincide com ataque da base/bumbo/caixa
    tonic_pc: int
    color: np.ndarray           # (P,) bool: altura é nota característica da escala do solo
    triplet: bool               # grade da família das tercinas (3, 6, 12...) ou binária (2, 4, 8...)
    swing: float                # swing aplicado ao solo (só na grade binária)
    scale_pc: np.ndarray        # (12,) bool: classe de altura pertence à escala do solo
    ct_pc: np.ndarray           # (n_slots, 12) bool: classe de altura é nota do acorde
    avoid_pc: np.ndarray        # (n_slots, 12) bool: classe de altura colide com o acorde
    lick_p: np.ndarray          # (len(LICKS),) probabilidade de sortear cada lick neste estilo
    lick_len: np.ndarray        # (len(LICKS),) tamanho de cada lick em slots
    lick_aff: np.ndarray        # (len(LICKS),) afinidade de cada lick com o gênero da música

    @property
    def slots_per_bar(self) -> int:
        return BEATS_PER_BAR * self.subdiv

    @property
    def n_bars(self) -> int:
        return self.n_slots // self.slots_per_bar

    @property
    def n_pitches(self) -> int:
        return len(self.pitches)

    @property
    def basic(self) -> int:
        """Slots por colcheia (ou por tercina de colcheia): a figura "simples" da grade."""
        return self.subdiv // (3 if self.triplet else 2)

    def chord_at_slot(self, slot: int) -> Chord:
        bar, s = divmod(slot, self.slots_per_bar)
        return self.cfg.chord_at(bar, s / self.subdiv)

    def slot_beats(self, slot: int) -> tuple[int, float]:
        """(compasso, tempo dentro do compasso) do início do slot."""
        bar, s = divmod(slot, self.slots_per_bar)
        return bar, s / self.subdiv


# Figura mais rápida do solo = quantos slots cabem em um tempo (semínima).
GRID_NAMES = {2: "colcheia", 4: "semicolcheia", 8: "fusa", 16: "semifusa",
              3: "tercina", 6: "sextina", 12: "tercina de fusa", 24: "sextina de fusa"}
BINARY_GRIDS = (2, 4, 8, 16)
TRIPLET_GRIDS = (3, 6, 12, 24)
AUTO_MAX_NPS = 13.0      # na grade automática: figura mais rápida até ~13 notas por segundo
PLAYABLE_NPS = 16.0      # acima disso, só shredders extremos (aviso na interface)


def triplet_feel(cfg: SongConfig) -> bool:
    """Ritmos com feel de tercina (slow blues, shuffle, swing >= 0.6) usam a família 3, 6, 12."""
    return cfg.preset.solo_subdiv == 3 or cfg.swing >= 0.6


def grid_options(cfg: SongConfig) -> tuple[int, ...]:
    return TRIPLET_GRIDS if triplet_feel(cfg) else BINARY_GRIDS


def auto_subdiv(cfg: SongConfig) -> int:
    """Figura mais fina que não passe de AUTO_MAX_NPS no andamento da música.

    Quanto mais lento o andamento, mais fina a figura: thrash a 180 bpm fica na
    semicolcheia (12 notas/s); doom a 64 bpm chega à fusa (8,5 notas/s).
    """
    opts = grid_options(cfg)
    ok = [g for g in opts if g * cfg.bpm / 60.0 <= AUTO_MAX_NPS]
    return max(ok) if ok else opts[0]


def build_space(cfg: SongConfig, subdiv: int | None = None) -> SoloSpace:
    """`subdiv`: figura mais rápida em slots por tempo (ver GRID_NAMES); None = automática."""
    subdiv = subdiv or auto_subdiv(cfg)
    triplet = subdiv % 3 == 0
    spb = BEATS_PER_BAR * subdiv
    n_slots = cfg.n_bars * spb
    scale = set(solo_scale(cfg))
    pitches = np.array([p for p in range(SOLO_LOW, SOLO_HIGH + 1) if p % 12 in scale])
    pcs = pitches % 12

    t = note_to_pc(cfg.tonic)
    color_pcs = {(t + i) % 12 for i in SCALE_COLOR[cfg.scale]}
    color = np.isin(pcs, list(color_pcs))

    ct_pc = np.zeros((n_slots, 12), bool)
    avoid_pc = np.zeros((n_slots, 12), bool)
    root_pc = np.zeros(n_slots, int)
    for slot in range(n_slots):
        bar, beat = divmod(slot, spb)
        chord = cfg.chord_at(bar, beat / subdiv)
        ct = set(chord.pitch_classes)
        root_pc[slot] = chord.root
        clash = {(c + 1) % 12 for c in ct} - ct - color_pcs
        if cfg.mode == "blues":
            clash -= {(chord.root + b) % 12 for b in _BLUE_NOTES}
        ct_pc[slot, list(ct)] = True
        avoid_pc[slot, list(clash)] = True
    chord_tone = ct_pc[:, pcs].T
    avoid = avoid_pc[:, pcs].T
    scale_pc = np.zeros(12, bool)
    scale_pc[list(scale)] = True

    # Ataques da base que caem na grade do solo (acompanhamento, bumbo e caixa).
    p = cfg.preset
    starts = [h.start for h in p.comp] + [d.start for d in p.drums if d.note in (KICK, SNARE)]
    bar_accent = np.zeros(spb, bool)
    for t in starts:
        s = t * subdiv
        if abs(s - round(s)) < 1e-6:
            bar_accent[int(round(s)) % spb] = True
    accent = np.tile(bar_accent, cfg.n_bars)

    return SoloSpace(cfg, subdiv, n_slots, pitches, chord_tone, avoid, root_pc, accent,
                     note_to_pc(cfg.tonic), color, triplet, 0.5 if triplet else cfg.swing,
                     scale_pc, ct_pc, avoid_pc, lick_probabilities(cfg, subdiv, triplet),
                     np.array([int(np.ceil(lk.beats * subdiv - 1e-9)) for lk in LICKS]),
                     np.array([lk.weight(cfg.preset.genre) for lk in LICKS]))


def lick_probabilities(cfg: SongConfig, subdiv: int, triplet: bool) -> np.ndarray:
    """Chance de sortear cada lick: afinidade com o gênero x compatibilidade com a grade."""
    w = []
    for lk in LICKS:
        x = lk.weight(cfg.preset.genre)
        if lk.triplet and not triplet:
            x *= 0.3        # lick em tercinas numa grade binária fica torto (o contrário
                            # é resolvido aplicando swing ao lick, ver write_lick)
        if lk.min_dur * subdiv < 0.99:
            x *= 0.2        # lick mais rápido do que a figura mais rápida permitida
        w.append(x)
    w = np.array(w)
    return w / w.sum()


# ------------------------------------------------------------------ decodificação

@dataclass
class Decoded:
    onset: np.ndarray    # slot de início de cada nota
    dur: np.ndarray      # duração em slots
    idx: np.ndarray      # índice em space.pitches
    pitch: np.ndarray    # altura MIDI
    dyad: np.ndarray     # distância da nota dupla em semitons (0 = sem nota dupla)
    art: np.ndarray      # articulação

    def __len__(self):
        return len(self.onset)


def as_matrix(genome: np.ndarray) -> np.ndarray:
    """Aceita também genomas antigos (só a linha de melodia)."""
    if genome.ndim == 2:
        return genome
    G = np.zeros((N_ROWS, len(genome)), int)
    G[MEL] = genome
    return G


def decode(genome: np.ndarray, space: SoloSpace) -> Decoded:
    """Genoma -> lista de notas (onset, duração, altura, nota dupla, articulação).
    HOLD após pausa vira pausa."""
    G = as_matrix(genome)
    genome = G[MEL]
    is_note = genome >= 0
    onset = np.flatnonzero(is_note)
    if len(onset) == 0:
        e = np.array([], int)
        return Decoded(e, e, e, e, e, e)
    # Cada nota dura até o próximo gene que não seja HOLD.
    not_hold = np.flatnonzero(genome != HOLD)
    nxt = np.searchsorted(not_hold, onset, side="right")
    end = np.where(nxt < len(not_hold), not_hold[np.minimum(nxt, len(not_hold) - 1)],
                   len(genome))
    idx = genome[onset]
    return Decoded(onset, end - onset, idx, space.pitches[idx], G[DYAD, onset], G[ART, onset])


# ------------------------------------------------------------------ inicialização

def beat_cells(space: SoloSpace) -> list[int]:
    """Quantidades de ataques por tempo que dividem a grade igualmente (ex.: 1, 2, 4, 8)."""
    return [k for k in range(1, space.subdiv + 1) if space.subdiv % k == 0]


def write_cell(g: np.ndarray, space: SoloSpace, beat: int, k: int, pitches: list[int]):
    """Escreve no tempo `beat` uma figura de `k` ataques igualmente espaçados."""
    sub = space.subdiv
    seg = np.full(sub, HOLD)
    step = sub // k
    seg[::step] = pitches[:k]
    g[beat * sub:(beat + 1) * sub] = seg


def scale_run(start: int, length: int, space: SoloSpace, rng: np.random.Generator) -> list[int]:
    """Escalada rápida: graus conjuntos numa direção, com uma eventual volta (bordadura)."""
    direction = rng.choice([-1, 1])
    out, cur = [], start
    for _ in range(length):
        out.append(int(cur))
        step = direction if rng.random() > 0.15 else -direction
        cur = int(np.clip(cur + step, 0, space.n_pitches - 1))
        if cur in (0, space.n_pitches - 1):
            direction = -direction
    return out


def nearest_index(space: SoloSpace, pitch: float) -> int:
    return int(np.argmin(np.abs(space.pitches - pitch)))


def previous_pitch(G: np.ndarray, slot: int, space: SoloSpace) -> int:
    prev = G[MEL, :slot][G[MEL, :slot] >= 0]
    return int(space.pitches[prev[-1]]) if len(prev) else int(np.median(space.pitches))


def lick_slots(lick: Lick, space: SoloSpace) -> int:
    return int(np.ceil(lick.beats * space.subdiv - 1e-9))


def write_lick(G: np.ndarray, space: SoloSpace, lick_idx: int, start_beat: int):
    """Escreve o lick `lick_idx` a partir do tempo `start_beat`, em todas as linhas.

    A oitava é a que fica mais perto da nota anterior do solo. Cada altura é ajustada à
    nota mais próxima da escala do solo (o lick "se adapta" ao modo escolhido); a nota
    dupla mantém a distância escrita (pode ser cromática, como no turnaround) ou, se for
    diatônica, escolhe a 3ª/6ª que pertence à escala. O ritmo é quantizado na grade.
    """
    lick = LICKS[lick_idx]
    sub = space.subdiv
    s0 = start_beat * sub
    s1 = min(s0 + lick_slots(lick, space), space.n_slots)
    chord = space.chord_at_slot(s0)
    ref = chord.root if lick.ref == "chord" else space.tonic_pc
    scale_pcs = set(np.flatnonzero(space.scale_pc).tolist())
    events = realize_lick(lick, ref, scale_pcs, chord, low=int(space.pitches[0]),
                          high=int(space.pitches[-1]), near=previous_pitch(G, s0, space))
    G[MEL, s0:s1] = REST
    G[DYAD, s0:s1] = 0
    G[ART, s0:s1] = 0
    G[SRC, s0:s1] = lick_idx + 1
    # Lick escrito em colcheias retas numa grade de tercinas (shuffle, slow blues): toca
    # swingado, como um guitarrista faria.
    swung = space.triplet and not lick.triplet

    def pos(beat: float) -> float:
        return swing_time(beat, 2 / 3) if swung else beat

    last = -1
    for ev in events:
        slot = s0 + int(round(pos(ev.start) * sub))
        if slot <= last or slot >= s1:
            continue            # duas notas caíram no mesmo slot: a grade é grossa demais
        dur = max(1, int(round((pos(ev.start + ev.dur) - pos(ev.start)) * sub)))
        G[MEL, slot] = nearest_index(space, ev.pitch)
        G[MEL, slot + 1:min(slot + dur, s1)] = HOLD
        snapped = int(space.pitches[G[MEL, slot]])
        G[DYAD, slot] = 0 if ev.dyad is None else ev.dyad - ev.pitch
        if ev.dyad is not None and snapped != ev.pitch:
            # A nota principal mudou ao entrar na escala. Nota dupla diatônica: refaz a
            # 3ª/6ª a partir da nova nota. Exata: mantém a altura escrita se ela é da
            # escala (a nota presa não muda); senão, mantém a distância.
            spec = next(n.dyad for n in lick.notes if abs(ev.start - n.start) < 1e-9)
            if isinstance(spec, str):
                G[DYAD, slot] = resolve_dyad(snapped, spec, scale_pcs) - snapped
            elif space.scale_pc[ev.dyad % 12] and ev.dyad != snapped:
                G[DYAD, slot] = ev.dyad - snapped
        G[ART, slot] = ev.art
        last = slot


def insert_random_lick(G: np.ndarray, space: SoloSpace, rng: np.random.Generator,
                       bar: int | None = None) -> bool:
    """Sorteia um lick pela afinidade com o estilo e o escreve num tempo aleatório
    (dentro do compasso `bar`, se dado). Devolve False se não couber."""
    idx = int(rng.choice(len(LICKS), p=space.lick_p))
    beats = int(np.ceil(LICKS[idx].beats - 1e-9))
    n_beats = space.n_slots // space.subdiv
    if bar is None:
        lo, hi = 0, n_beats - beats
    else:
        lo, hi = bar * BEATS_PER_BAR, min(bar * BEATS_PER_BAR + 3, n_beats - beats)
    if hi < lo:
        return False
    write_lick(G, space, idx, int(rng.integers(lo, hi + 1)))
    return True


def random_genome(space: SoloSpace, rng: np.random.Generator, method: str = "walk",
                  density: float = 1.6, shred: float = 0.2,
                  lick_rate: float = 0.3) -> np.ndarray:
    """Indivíduo aleatório, construído tempo a tempo.

    Cada tempo recebe uma figura: pausa, sustentação da nota anterior, 1-3 ataques simples
    ou uma rajada na grade mais fina (com probabilidade ligada a `shred`). As alturas seguem
    um passeio aleatório na escala ("walk") ou são sorteadas uniformemente ("uniform").
    `density` (notas por tempo) ajusta a chance de figuras mais cheias. Depois, cada
    compasso recebe um lick do estilo com probabilidade `lick_rate`.
    """
    n_beats = space.n_slots // space.subdiv
    P = space.n_pitches
    g = np.full(space.n_slots, HOLD)
    cells = beat_cells(space)
    simple = [k for k in cells if k <= 3]
    fast = [k for k in cells if k >= 4] or [cells[-1]]
    cur = int(rng.integers(P // 4, 3 * P // 4))
    p_rest = 0.10
    p_hold = max(0.05, 0.3 - 0.08 * density)

    def next_pitch():
        nonlocal cur
        if method == "uniform":
            cur = int(rng.integers(P))
        else:
            cur = int(np.clip(cur + rng.choice([-2, -1, 0, 1, 2], p=[.15, .3, .1, .3, .15]),
                              0, P - 1))
        return cur

    for b in range(n_beats):
        r = rng.random()
        if r < p_rest:
            g[b * space.subdiv] = REST
        elif r < p_rest + p_hold and b > 0:
            pass                                    # tudo HOLD: sustenta o que vinha
        elif rng.random() < shred * 0.6:
            k = int(rng.choice(fast))
            run = scale_run(cur, k, space, rng) if method == "walk" else \
                [int(x) for x in rng.integers(0, P, size=k)]
            cur = run[-1]
            write_cell(g, space, b, k, run)
        else:
            weights = np.array([1.0 / (1 + abs(k - density)) for k in simple])
            k = int(rng.choice(simple, p=weights / weights.sum()))
            write_cell(g, space, b, k, [next_pitch() for _ in range(k)])
    if g[0] == HOLD:
        g[0] = REST
    G = np.zeros((N_ROWS, space.n_slots), int)
    G[MEL] = g
    # Licks: cada compasso tem chance `lick_rate` de receber um lick do estilo.
    for bar in range(space.n_bars):
        if rng.random() < lick_rate:
            insert_random_lick(G, space, rng, bar)
    return G
