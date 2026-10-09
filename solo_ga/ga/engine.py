"""Laço evolutivo: população, elitismo, critério de parada e histórico de fitness."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..config import SongConfig
from ..licks import LICKS
from ..rhythms import SOLO_SPEED
from .fitness import FitnessParams, evaluate
from .genome import MEL, SRC, SoloSpace, build_space, lick_slots, random_genome
from .operators import crossover, mutate, tournament


@dataclass
class GAParams:
    pop_size: int = 120
    generations: int = 300
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3        # probabilidade de cada uma das (até 4) mutações por filho
    tournament_k: int = 3
    elitism: int = 2
    immigrants: float = 0.05          # fração da população trocada por indivíduos novos a
                                      # cada geração (imigrantes aleatórios: mantém diversidade)
    patience: int = 80                # para se o melhor não melhora por N gerações
    init: str = "walk"                # "walk" ou "uniform"
    lick_rate: float = 0.3            # chance de lick por compasso na população inicial;
                                      # também escala a frequência da mutação de lick
    grid: int | None = None           # figura mais rápida (slots por tempo: 4 = semicolcheia,
                                      # 8 = fusa...); None = automática pelo andamento
    seed: int = 42
    fitness: FitnessParams = field(default_factory=FitnessParams)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GAResult:
    best: np.ndarray
    best_fitness: float
    breakdown: dict[str, float]
    history: dict[str, list[float]]   # best / mean / worst / diversity por geração
    generations_run: int
    stop_reason: str
    elapsed_sec: float
    space: SoloSpace
    fitness_params: FitnessParams     # com densidade/shred já resolvidos


def diversity(pop: np.ndarray) -> float:
    """Distância de Hamming média entre pares de indivíduos, na linha de melodia (0 a 1)."""
    mel = pop[:, MEL]
    n = len(mel)
    if n < 2:
        return 0.0
    diff = (mel[:, None, :] != mel[None, :, :]).mean(axis=2)
    return float(diff.sum() / (n * (n - 1)))


def resolve_fitness(cfg: SongConfig, fp: FitnessParams) -> FitnessParams:
    """Preenche densidade e shred não informados com o perfil do estilo."""
    dens, shred = SOLO_SPEED[cfg.rhythm]
    return dataclasses.replace(fp, density=dens if fp.density is None else fp.density,
                               shred=shred if fp.shred is None else fp.shred)


def lick_usage(G: np.ndarray, space: SoloSpace) -> list[dict]:
    """Licks presentes no genoma: trechos contínuos com a mesma marca de origem.
    `intact` = fração do lick original que sobreviveu sem alteração."""
    out, src, i = [], G[SRC], 0
    while i < len(src):
        if src[i] == 0:
            i += 1
            continue
        j = i
        while j < len(src) and src[j] == src[i]:
            j += 1
        lk = LICKS[src[i] - 1]
        out.append({"lick": lk.name, "key": lk.key, "genre": lk.genre,
                    "beat": i / space.subdiv,
                    "intact": min(1.0, (j - i) / lick_slots(lk, space))})
        i = j
    return out


ProgressFn = Callable[[int, dict[str, list[float]], np.ndarray], None]


def run_ga(cfg: SongConfig, params: GAParams, progress: ProgressFn | None = None,
           progress_every: int = 5) -> GAResult:
    rng = np.random.default_rng(params.seed)
    space = build_space(cfg, params.grid)
    fp = resolve_fitness(cfg, params.fitness)

    pop = np.stack([random_genome(space, rng, params.init, fp.density, fp.shred,
                                  params.lick_rate) for _ in range(params.pop_size)])
    fit = np.array([evaluate(g, space, fp)[0] for g in pop])

    hist = {"best": [], "mean": [], "worst": [], "diversity": []}
    best_so_far, stale, stop = -np.inf, 0, "gerações"
    t0 = time.perf_counter()

    for gen in range(params.generations):
        hist["best"].append(float(fit.max()))
        hist["mean"].append(float(fit.mean()))
        hist["worst"].append(float(fit.min()))
        hist["diversity"].append(diversity(pop))
        if progress and gen % progress_every == 0:
            progress(gen, hist, pop[np.argmax(fit)])

        if fit.max() > best_so_far + 1e-6:
            best_so_far, stale = fit.max(), 0
        else:
            stale += 1
            if stale >= params.patience:
                stop = "estagnação"
                break

        # Elitismo: os melhores passam intactos.
        elite = np.argsort(fit)[-params.elitism:] if params.elitism else np.array([], int)
        children = [pop[i].copy() for i in elite]

        # Imigrantes aleatórios: entram no lugar dos últimos filhos, criados como na
        # população inicial, para repor material genético que a seleção eliminou.
        n_imm = min(int(round(params.immigrants * params.pop_size)),
                    params.pop_size - len(children))
        n_needed = params.pop_size - len(children) - n_imm
        parents = tournament(fit, params.tournament_k, n_needed + 1, rng)
        for i in range(0, n_needed, 2):
            a, b = pop[parents[i]], pop[parents[i + 1]]
            if rng.random() < params.crossover_rate:
                c1, c2 = crossover(a, b, space, rng)
            else:
                c1, c2 = a.copy(), b.copy()
            children.append(mutate(c1, space, params.mutation_rate, rng,
                                   lick_rate=params.lick_rate))
            if len(children) < params.pop_size - n_imm:
                children.append(mutate(c2, space, params.mutation_rate, rng,
                                       lick_rate=params.lick_rate))

        children += [random_genome(space, rng, params.init, fp.density, fp.shred,
                                   params.lick_rate) for _ in range(n_imm)]
        pop = np.stack(children)
        fit = np.array([evaluate(g, space, fp)[0] for g in pop])

    best_i = int(np.argmax(fit))
    total, breakdown = evaluate(pop[best_i], space, fp)
    if progress:
        progress(len(hist["best"]), hist, pop[best_i])
    return GAResult(pop[best_i].copy(), total, breakdown, hist, len(hist["best"]), stop,
                    time.perf_counter() - t0, space, fp)
