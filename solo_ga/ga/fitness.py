"""Função de fitness heurística: soma ponderada de critérios de teoria musical e fraseado.

Cada critério devolve um valor em [0, 1]; a fitness é a média ponderada pelos pesos.
Todos os critérios olham para a mesma progressão e o mesmo preset rítmico da base,
então o solo é avaliado "em contexto".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .genome import SRC, Decoded, SoloSpace, as_matrix, decode

# (nome, rótulo para UI/resumo, peso padrão)
CRITERIA: list[tuple[str, str, float]] = [
    ("harmony", "Notas do acorde nos tempos fortes e notas longas", 3.0),
    ("consonance", "Evitar notas que colidem com o acorde", 2.0),
    ("intervals", "Intervalos pequenos predominam", 2.0),
    ("density", "Densidade média (notas por tempo) próxima do alvo", 1.5),
    ("shred", "Rajadas rápidas (fração de notas em runs) próxima do alvo", 1.5),
    ("rhythm_coherence", "Coerência rítmica: figuras regulares, sem ataques soltos", 1.5),
    ("phrasing", "Frases com respiração (pausas), sem notas picotadas", 1.5),
    ("motif", "Repetição rítmica entre compassos (pergunta e resposta)", 1.5),
    ("ending", "Final resolvendo na tônica com nota longa", 1.5),
    ("idiom", "Idioma do estilo: trechos vindos de licks do gênero, preservados", 1.5),
]
CRITERIA_LABELS = {k: lbl for k, lbl, _ in CRITERIA}


def default_weights() -> dict[str, float]:
    return {k: w for k, _, w in CRITERIA}


@dataclass
class FitnessParams:
    weights: dict[str, float] = field(default_factory=default_weights)
    density: float | None = None  # notas por tempo (média do solo); None -> padrão do ritmo
    shred: float | None = None    # fração alvo de notas em rajadas; None -> padrão do ritmo
    run_min_len: int = 4          # notas rápidas seguidas para contar como rajada
    lick_coverage: float = 0.35   # fração do solo coberta por licks do estilo a partir da
                                  # qual o critério de idioma satura
    rest_ratio: float = 0.12     # fração alvo do tempo em pausa
    min_rest_beats: float = 0.5  # pausa menor que isso entre duas notas "quebra" a frase
    max_phrase_beats: float = 8  # frase mais longa aceitável sem respiro (em tempos)
    motif_similarity: float = 0.6   # similaridade rítmica (Jaccard) alvo entre compassos pareados


def _closeness(x: float, target: float) -> float:
    """1 quando x == target, caindo linearmente até 0 a uma distância de `target`."""
    return float(max(0.0, 1.0 - abs(x - target) / max(target, 1e-9)))


def _interval_score(semitones: np.ndarray) -> np.ndarray:
    # 0: nota repetida (aceitável); 1-4: graus conjuntos/terças; 5-7: saltos médios;
    # 8-12: saltos grandes; >12: mais de uma oitava.
    return np.select([semitones == 0, semitones <= 4, semitones <= 7, semitones <= 12],
                     [0.6, 1.0, 0.6, 0.2], 0.0)


def idiom_score(src: np.ndarray, space: SoloSpace, target: float) -> float:
    """Cobertura do solo por licks preservados, ponderada pela afinidade com o gênero.

    Cada trecho contínuo com a mesma marca de lick conta se tiver pelo menos metade do
    lick original; vale (tamanho do trecho x afinidade do lick com o gênero). Um lick de
    outro estilo (afinidade ~0,05) quase não conta; um do próprio estilo vale integralmente.
    """
    if target <= 0:
        return 1.0
    edges = np.flatnonzero(np.diff(src)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(src)]])
    ids = src[starts]
    keep = ids > 0
    if not keep.any():
        return 0.0
    lick = ids[keep] - 1
    length = (ends - starts)[keep]
    intact = length / space.lick_len[lick]
    ok = intact >= 0.5
    cov = float(np.sum(length[ok] * space.lick_aff[lick][ok])) / space.n_slots
    return min(1.0, cov / target)


def evaluate(genome: np.ndarray, space: SoloSpace, fp: FitnessParams,
             notes: Decoded | None = None) -> tuple[float, dict[str, float]]:
    """Retorna (fitness total, pontuação de cada critério)."""
    n = notes or decode(genome, space)
    zero = {k: 0.0 for k in fp.weights}
    if len(n) < 4:
        return 0.0, zero

    spb, sub = space.slots_per_bar, space.subdiv
    ct = space.chord_tone[n.idx, n.onset]
    av = space.avoid[n.idx, n.onset]
    s = {}

    # Harmonia: notas do acorde nos tempos (onset no início do tempo) e nas notas longas.
    on_beat = (n.onset % sub) == 0
    long_ = n.dur >= sub
    h_beat = ct[on_beat].mean() if on_beat.any() else 0.0
    h_long = np.average(ct[long_], weights=n.dur[long_]) if long_.any() else 0.0
    s["harmony"] = 0.5 * h_beat + 0.5 * h_long

    # Consonância: tempo (ponderado pela duração) gasto em notas que colidem com o acorde.
    # A nota dupla também conta: ela não pode colidir com o acorde e precisa ser da escala
    # ou do acorde (notas cromáticas curtas, como no turnaround, custam pouco).
    has_dyad = n.dyad != 0
    dpc = (n.pitch + n.dyad) % 12
    dyad_bad = has_dyad & (space.avoid_pc[n.onset, dpc] |
                           (~space.scale_pc[dpc] & ~space.ct_pc[n.onset, dpc]))
    s["consonance"] = 1.0 - float(np.average(av | dyad_bad, weights=n.dur))

    # Intervalos melódicos.
    s["intervals"] = float(_interval_score(np.abs(np.diff(n.pitch))).mean())

    # Densidade: notas por tempo no solo todo (pode variar entre compassos: rajadas
    # convivem com notas longas).
    n_beats = space.n_slots / sub
    s["density"] = _closeness(len(n) / n_beats, fp.density)

    # Shred: notas de semicolcheia (ou mais rápidas) em sequências de >= run_min_len.
    dur_beats = n.dur / sub
    fast = dur_beats <= 0.26
    edges_f = np.diff(np.concatenate([[0], fast.astype(int), [0]]))
    f_start, f_end = np.flatnonzero(edges_f == 1), np.flatnonzero(edges_f == -1)
    run_len = f_end - f_start
    in_runs = run_len[run_len >= fp.run_min_len].sum() / len(n)
    s["shred"] = _closeness(in_runs, fp.shred) if fp.shred > 0 else 1.0 - in_runs

    # Coerência rítmica: o ataque cai na grade simples (colcheia/tercina) ou faz parte de
    # uma sequência regular (>= 3 notas seguidas com a mesma duração, ex.: uma rajada).
    on_simple = (n.onset % space.basic) == 0
    same_prev = np.concatenate([[False], n.dur[1:] == n.dur[:-1]])
    same_next = np.concatenate([n.dur[:-1] == n.dur[1:], [False]])
    regular = np.zeros(len(n), bool)
    regular[1:-1] = same_prev[1:-1] & same_next[1:-1]
    regular[1:] |= regular[:-1]
    regular[:-1] |= regular[1:].copy()
    s["rhythm_coherence"] = float(np.mean(on_simple | regular))

    # Fraseado: proporção de pausa e tamanho das frases contínuas.
    sounding = np.zeros(space.n_slots, bool)
    for o, d in zip(n.onset, n.dur):
        sounding[o:o + d] = True
    rest_frac = 1.0 - sounding.mean()
    # Frases = trechos contínuos soando; mede quanto excedem o limite.
    edges = np.diff(np.concatenate([[0], sounding.astype(int), [0]]))
    runs = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / sub
    too_long = np.clip(runs - fp.max_phrase_beats, 0, None).sum() / max(runs.sum(), 1e-9)
    # Buracos curtos entre notas soam como notas picotadas; os que vêm de licks (ex.:
    # chicken picking) são intencionais e não contam.
    ends = n.onset + n.dur
    gap = n.onset[1:] - ends[:-1]
    real = gap > 0
    from_lick = as_matrix(genome)[SRC][np.minimum(ends[:-1], space.n_slots - 1)] > 0
    short = real & (gap < fp.min_rest_beats * sub) & ~from_lick
    choppy = short.sum() / max(real.sum(), 1)
    s["phrasing"] = (0.4 * _closeness(rest_frac, fp.rest_ratio) + 0.3 * (1.0 - too_long)
                     + 0.3 * (1.0 - choppy))

    # Motivo: compassos pareados (0-1, 2-3, ...) com padrão rítmico parecido.
    onset_mask = np.zeros(space.n_slots, bool)
    onset_mask[n.onset] = True
    bars = onset_mask.reshape(space.n_bars, spb)
    pairs = [(i, i + 1) for i in range(0, space.n_bars - 1, 2)]
    if pairs:
        sims = [np.sum(bars[a] & bars[b]) / max(np.sum(bars[a] | bars[b]), 1)
                for a, b in pairs]
        s["motif"] = float(np.mean([_closeness(x, fp.motif_similarity) for x in sims]))
    else:
        s["motif"] = 1.0

    # Final: última nota na tônica (ou ao menos nota do acorde), longa, no último compasso.
    last_pc = n.pitch[-1] % 12
    e_pitch = 1.0 if last_pc == space.tonic_pc else (0.5 if ct[-1] else 0.0)
    e_len = min(n.dur[-1] / (2 * sub), 1.0)
    e_pos = 1.0 if n.onset[-1] >= space.n_slots - spb else 0.0
    s["ending"] = 0.5 * e_pitch + 0.25 * e_len + 0.25 * e_pos

    s["idiom"] = idiom_score(as_matrix(genome)[SRC], space, fp.lick_coverage)

    w = fp.weights
    total = sum(w[k] * s[k] for k in w) / max(sum(w.values()), 1e-9)
    return float(total), {k: float(v) for k, v in s.items()}
