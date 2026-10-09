"""Gráfico da evolução do fitness (obrigatório no resumo)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_fitness(histories: dict[str, dict[str, list[float]]], path: str | Path,
                 title: str = "Evolução do fitness", show_band: bool = True) -> Path:
    """`histories`: {rótulo: {"best": [...], "mean": [...], "worst": [...]}}.

    Uma curva de melhor indivíduo por rótulo (linha cheia) e a média da população
    (tracejada); com um único rótulo, a faixa pior-melhor fica sombreada.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    for i, (label, h) in enumerate(histories.items()):
        gens = range(len(h["best"]))
        line, = ax.plot(gens, h["best"], lw=1.8, label=f"{label} (melhor)")
        ax.plot(gens, h["mean"], lw=1.1, ls="--", color=line.get_color(),
                label=f"{label} (média)")
        if show_band and len(histories) == 1:
            ax.fill_between(gens, h["worst"], h["best"], color=line.get_color(), alpha=0.12,
                            lw=0, label="faixa pior-melhor")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
