"""Operadores genéticos: seleção por torneio, cruzamento em fronteira de compasso e mutações.

As mutações "de melodia" recebem só a linha MEL do genoma (vetor 1D, alterado no lugar);
as demais recebem a matriz inteira (MEL, DYAD, ART, SRC). `mutate` cuida da consistência
entre as linhas depois de cada mutação.
"""

from __future__ import annotations

import numpy as np

from ..licks import AUTO, BEND1, BEND2, RING, STRAIGHT, resolve_dyad
from .genome import (ART, DYAD, HOLD, MEL, REST, SRC, SoloSpace, beat_cells,
                     insert_random_lick, scale_run, write_cell)


# ------------------------------------------------------------------ seleção

def tournament(fitness: np.ndarray, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Seleciona `n` índices: cada um é o melhor de `k` indivíduos sorteados."""
    cand = rng.integers(0, len(fitness), size=(n, k))
    return cand[np.arange(n), np.argmax(fitness[cand], axis=1)]


# ------------------------------------------------------------------ cruzamento

def crossover(a: np.ndarray, b: np.ndarray, space: SoloSpace,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cruzamento de dois pontos com cortes nas fronteiras de compasso.

    Cortar só entre compassos preserva frases inteiras dentro de cada compasso, em vez de
    partir uma nota ao meio.
    """
    nb, spb = space.n_bars, space.slots_per_bar
    if nb < 2:
        return a.copy(), b.copy()
    i, j = sorted(rng.choice(np.arange(1, nb + 1), size=2, replace=False))
    i, j = i * spb, j * spb
    c1, c2 = a.copy(), b.copy()
    c1[:, i:j], c2[:, i:j] = b[:, i:j], a[:, i:j]
    return c1, c2


# ------------------------------------------------------------------ mutações

def _note_slots(g: np.ndarray) -> np.ndarray:
    return np.flatnonzero(g >= 0)


def mut_pitch_step(g, space, rng):
    """Move uma nota 1 ou 2 graus da escala para cima/baixo."""
    idx = _note_slots(g)
    if len(idx):
        s = rng.choice(idx)
        g[s] = np.clip(g[s] + rng.choice([-2, -1, 1, 2]), 0, space.n_pitches - 1)


def mut_random_gene(g, space, rng):
    """Troca um gene qualquer por nota, pausa ou sustentação aleatória."""
    s = rng.integers(len(g))
    r = rng.random()
    g[s] = REST if r < 0.1 else HOLD if r < 0.45 else rng.integers(space.n_pitches)


def mut_rhythm(g, space, rng):
    """Altera a duração: transforma um ataque em sustentação ou vice-versa."""
    s = rng.integers(1, len(g))
    if g[s] >= 0:
        g[s] = HOLD
    elif g[s] == HOLD:
        prev = g[:s][g[:s] >= 0]
        base = prev[-1] if len(prev) else rng.integers(space.n_pitches)
        g[s] = np.clip(base + rng.choice([-1, 1]), 0, space.n_pitches - 1)


def mut_copy_bar(G, space, rng):
    """Copia um compasso (todas as linhas) sobre outro, transpondo de -2 a +2 graus:
    cria motivos e sequências, e também replica licks."""
    nb, spb = space.n_bars, space.slots_per_bar
    if nb < 2:
        return
    a, b = rng.choice(nb, size=2, replace=False)
    src = G[:, a * spb:(a + 1) * spb].copy()
    shift = rng.integers(-2, 3)
    notes = src[MEL] >= 0
    src[MEL, notes] = np.clip(src[MEL, notes] + shift, 0, space.n_pitches - 1)
    G[:, b * spb:(b + 1) * spb] = src


def mut_invert(g, space, rng):
    """Espelha o contorno das notas de um compasso em torno da primeira nota (inversão)."""
    spb = space.slots_per_bar
    bar = rng.integers(space.n_bars)
    seg = g[bar * spb:(bar + 1) * spb]
    notes = seg >= 0
    if notes.sum() >= 2:
        axis = seg[notes][0]
        seg[notes] = np.clip(2 * axis - seg[notes], 0, space.n_pitches - 1)


def _pitch_before(g: np.ndarray, slot: int, space: SoloSpace, rng) -> int:
    prev = g[:slot][g[:slot] >= 0]
    return int(prev[-1]) if len(prev) else int(rng.integers(space.n_pitches))


def mut_run(g, space, rng):
    """Insere uma rajada (escalada em graus conjuntos) na grade mais fina, em 1 ou 2 tempos."""
    sub = space.subdiv
    n_beats = len(g) // sub
    fast = [k for k in beat_cells(space) if k >= 4] or [sub]
    k = int(rng.choice(fast))
    span = int(rng.choice([1, 2])) if n_beats > 1 else 1
    b0 = int(rng.integers(0, n_beats - span + 1))
    cur = _pitch_before(g, b0 * sub, space, rng)
    run = scale_run(cur, k * span, space, rng)
    for i in range(span):
        write_cell(g, space, b0 + i, k, run[i * k:(i + 1) * k])


def mut_cell(g, space, rng):
    """Reescreve o ritmo de um tempo com uma figura regular, reaproveitando as alturas."""
    sub = space.subdiv
    b = int(rng.integers(0, len(g) // sub))
    seg = g[b * sub:(b + 1) * sub]
    old = [int(x) for x in seg[seg >= 0]] or [_pitch_before(g, b * sub, space, rng)]
    k = int(rng.choice(beat_cells(space)))
    pitches = [old[i % len(old)] for i in range(k)]
    if k > len(old):     # completa com passos de escala a partir da última nota
        pitches = old + scale_run(old[-1], k - len(old) + 1, space, rng)[1:]
    write_cell(g, space, b, k, pitches)


def mut_lick(G, space, rng):
    """Insere um lick do estilo (sorteado pela afinidade com o gênero) num tempo aleatório."""
    insert_random_lick(G, space, rng)


# Notas duplas mais comuns na guitarra: oitava abaixo, 4ª abaixo/acima, 3ªs e 6ªs diatônicas.
DYAD_CHOICES = [-12, -5, 5, "+3d", "-3d", "-6d"]


def mut_dyad(G, space, rng):
    """Acrescenta, troca ou remove a nota dupla de um ataque."""
    idx = np.flatnonzero(G[MEL] >= 0)
    if not len(idx):
        return
    s = rng.choice(idx)
    if G[DYAD, s] != 0 and rng.random() < 0.5:
        G[DYAD, s] = 0
        return
    p = int(space.pitches[G[MEL, s]])
    spec = DYAD_CHOICES[rng.integers(len(DYAD_CHOICES))]
    scale_pcs = set(np.flatnonzero(space.scale_pc).tolist())
    G[DYAD, s] = resolve_dyad(p, spec, scale_pcs) - p


def mut_art(G, space, rng):
    """Troca a articulação de um ataque (bend de tom/semitom, sem bend, deixa soar)."""
    idx = np.flatnonzero(G[MEL] >= 0)
    if len(idx):
        G[ART, rng.choice(idx)] = rng.choice([AUTO, AUTO, BEND2, BEND1, STRAIGHT, RING])


# (função, recebe a matriz inteira?)
MUTATIONS = {
    "pitch_step": (mut_pitch_step, False),
    "random_gene": (mut_random_gene, False),
    "rhythm": (mut_rhythm, False),
    "copy_bar": (mut_copy_bar, True),
    "invert": (mut_invert, False),
    "run": (mut_run, False),
    "cell": (mut_cell, False),
    "dyad": (mut_dyad, True),
    "art": (mut_art, True),
    "lick": (mut_lick, True),
}
# Probabilidade relativa de cada mutação, dado que uma mutação ocorre. O peso de "lick"
# é definido por `lick_rate` em `mutate`.
MUTATION_WEIGHTS = {"pitch_step": 0.26, "random_gene": 0.07, "rhythm": 0.1, "copy_bar": 0.1,
                    "invert": 0.05, "run": 0.12, "cell": 0.14, "dyad": 0.08, "art": 0.04}
# Mutações que escrevem todas as linhas de propósito (não perdem a marca de origem).
_KEEPS_SOURCE = {"copy_bar", "lick"}


def mutate(G: np.ndarray, space: SoloSpace, rate: float, rng: np.random.Generator,
           max_ops: int = 4, lick_rate: float = 0.3, lick_protect: float = 0.7) -> np.ndarray:
    """Aplica até `max_ops` mutações; cada tentativa ocorre com probabilidade `rate`.

    Depois de cada mutação:
      - onde um ataque virou pausa/sustentação (ou o contrário), a nota dupla e a
        articulação daquele slot são zeradas (só valem para ataques);
      - slots alterados perdem a marca de lick (SRC), exceto nas mutações que copiam/
        escrevem licks inteiros;
      - proteção: se a mutação alterou um trecho de lick, ela é desfeita com
        probabilidade `lick_protect` (os licks podem variar, mas devagar).
    """
    G = G.copy()
    weights = dict(MUTATION_WEIGHTS, lick=0.4 * lick_rate)
    names = list(weights)
    p = np.array([weights[n] for n in names])
    p /= p.sum()
    for _ in range(max_ops):
        if rng.random() >= rate:
            continue
        name = names[rng.choice(len(names), p=p)]
        fn, full = MUTATIONS[name]
        before = G.copy()
        fn(G if full else G[MEL], space, rng)
        changed = np.any(G[:SRC] != before[:SRC], axis=0)
        if not changed.any():
            continue
        if (name not in _KEEPS_SOURCE and (before[SRC, changed] > 0).any()
                and rng.random() < lick_protect):
            G[:] = before
            continue
        kind_changed = (G[MEL] >= 0) != (before[MEL] >= 0)
        stale = kind_changed & (G[MEL] < 0)
        G[DYAD, stale] = 0
        G[ART, stale] = 0
        if name not in _KEEPS_SOURCE:
            G[SRC, changed] = 0
    if G[MEL, 0] == HOLD:   # o primeiro gene não pode sustentar nada
        G[MEL, 0] = REST
    return G
