"""Experimentos do resumo ISMIR. Uso (na raiz do repo): python resumo/experimentos.py . resumo/exp"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO))

from solo_ga.config import SongConfig  # noqa: E402
from solo_ga.ga import GAParams, lick_usage, run_ga  # noqa: E402
from solo_ga.theory import preset_progression  # noqa: E402

# (progressão pronta ou acordes, tônica, ritmo, voltas, modo, escala do solo)
STYLES = {
    "blues": ("12-bar blues", "A", "shuffle", 2, None, "auto"),
    "thrash": ("i-VI-VII (metal)", "E", "thrash", 2, None, "auto"),
    "hardrock": ("i-VII-VI-VII (rock menor)", "E", "hard_rock", 2, None, "auto"),
    "funk": ("Em7 A7 Em7 A7 Em7 A7 Em7 A7", "E", "funk", 2, "minor", "auto"),
    "ballad": ("I-vi-IV-V (indie / doo-wop)", "C", "ballad", 2, None, "ionian"),
}


def cfg_for(style):
    prog, tonic, rhythm, ch, mode, scale = STYLES[style]
    from solo_ga.theory import PRESET_PROGRESSIONS
    if prog in PRESET_PROGRESSIONS:
        mode, prog = PRESET_PROGRESSIONS[prog]["mode"], preset_progression(prog, tonic)
    return SongConfig(prog, rhythm=rhythm, tonic=tonic, mode=mode, choruses=ch,
                      solo_scale=scale)


def job(spec):
    exp, style, seed, over = spec
    p = dataclasses.replace(GAParams(seed=seed), **over)
    r = run_ga(cfg_for(style), p)
    licks = [u for u in lick_usage(r.best, r.space) if u["intact"] >= 0.5]
    return {"exp": exp, "style": style, "seed": seed, "over": over,
            "fitness": r.best_fitness, "breakdown": r.breakdown, "gens": r.generations_run,
            "stop": r.stop_reason, "sec": r.elapsed_sec, "history": r.history,
            "n_licks": len(licks), "subdiv": r.space.subdiv, "n_slots": r.space.n_slots,
            "n_pitches": int(r.space.n_pitches)}


specs = []
for s in range(1, 6):
    specs.append(("base", "blues", s, {}))
for m in (0.1, 0.6):
    for s in range(1, 4):
        specs.append((f"mut{m}", "blues", s, {"mutation_rate": m}))
for s in range(1, 4):
    specs.append(("nolick", "blues", s, {"lick_rate": 0.0}))
    specs.append(("uniform", "blues", s, {"init": "uniform"}))
for st in ("thrash", "hardrock", "funk", "ballad"):
    for s in range(1, 4):
        specs.append(("style", st, s, {}))

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        res = pool.map(job, specs)
    (OUT / "results.json").write_text(json.dumps(res))
    for r in res:
        print(r["exp"], r["style"], r["seed"], f'{r["fitness"]:.4f}', r["gens"], r["stop"],
              f'{r["sec"]:.0f}s', r["n_licks"], r["subdiv"])
