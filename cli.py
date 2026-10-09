"""Linha de comando: gera os arquivos de forma reproduzível, sem interface.

Exemplos:
    python cli.py list
    python cli.py backing --preset "12-bar blues" --tonic A --rhythm shuffle
    python cli.py solo --preset "12-bar blues" --tonic A --rhythm shuffle --seed 1
    python cli.py solo --chords "Em Em C D" --mode minor --rhythm metal_gallop \\
        --gens 400 --mutation 0.5 --weight motif=3 --seed 7
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from solo_ga.config import SongConfig
from solo_ga.ga.genome import GRID_NAMES
from solo_ga.ga import CRITERIA, FitnessParams, GAParams, default_weights, lick_usage, run_ga
from solo_ga.midi_io import build_backing_midis, build_song_midis, write_midis
from solo_ga.plots import plot_fitness
from solo_ga.render import render_wav
from solo_ga.rhythms import PRESETS
from solo_ga.rhythms import SOLO_SCALES
from solo_ga.theory import (MODES, PRESET_PROGRESSIONS, SCALE_LABELS, SCALES,
                            preset_progression, scale_note_names)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _song_config(args) -> SongConfig:
    if args.chords:
        chords, mode = args.chords, args.mode or "major"
    else:
        chords = preset_progression(args.preset, args.tonic)
        mode = args.mode or PRESET_PROGRESSIONS[args.preset]["mode"]
    return SongConfig(chords, rhythm=args.rhythm, bpm=args.bpm, tonic=args.tonic, mode=mode,
                      choruses=args.choruses, swing=args.swing,
                      solo_scale=getattr(args, "scale", "auto"))


def _default_name(args, cfg: SongConfig) -> str:
    prog = _slug(args.preset) if not args.chords else "custom"
    return f"{prog}_{args.tonic}_{cfg.rhythm}_{cfg.bpm}"


def cmd_list(args):
    print("Ritmos (por gênero):")
    for k, p in PRESETS.items():
        print(f"  {p.genre:10s} {k:18s} {p.label} ({p.default_bpm} bpm) - {p.description}")
    print(f"\nEscalas do solo (tônica {args.tonic}):")
    for k in SCALES:
        print(f"  {k:18s} {SCALE_LABELS[k]:28s} {' '.join(scale_note_names(args.tonic, k))}")
    print("\nEscala automática por ritmo (maior / menor / blues):")
    for k, (ma, mi, bl) in SOLO_SCALES.items():
        print(f"  {k:18s} {ma} / {mi} / {bl}")
    print(f"\nProgressões prontas (tônica {args.tonic}):")
    for name in PRESET_PROGRESSIONS:
        print(f"  {name!r}: {preset_progression(name, args.tonic)}")
    print("\nCritérios da fitness (peso padrão):")
    for k, lbl, w in CRITERIA:
        print(f"  {k:14s} {w:4.1f}  {lbl}")


def cmd_backing(args):
    cfg = _song_config(args)
    out = Path(args.out) / (args.name or _default_name(args, cfg))
    midis = build_backing_midis(cfg)
    paths = write_midis(midis, out)
    print(f"Progressão: {cfg.progression}")
    print(f"Ritmo: {cfg.rhythm}  BPM: {cfg.bpm}  swing: {cfg.swing:.2f}  choruses: {cfg.choruses}")
    for k, p in paths.items():
        print(f"  MIDI {k:13s} -> {p}")
    if not args.no_audio:
        print(f"  WAV  base+bateria  -> {render_wav(midis['base_bateria'], out / 'base_bateria.wav')}")


def _parse_weights(items: list[str]) -> dict[str, float]:
    w = default_weights()
    for item in items or []:
        k, _, v = item.partition("=")
        if k not in w:
            raise SystemExit(f"Critério desconhecido: {k}. Opções: {', '.join(w)}")
        w[k] = float(v)
    return w


def _parse_grid(name: str) -> int | None:
    if name == "auto":
        return None
    by_name = {v.replace(" ", "_"): k for k, v in GRID_NAMES.items()}
    if name in by_name:
        return by_name[name]
    raise SystemExit(f"Figura desconhecida: {name}. Opções: auto, {', '.join(by_name)}")


def cmd_solo(args):
    cfg = _song_config(args)
    fp = FitnessParams(weights=_parse_weights(args.weight), density=args.density,
                       shred=args.shred, rest_ratio=args.rest_ratio,
                       lick_coverage=args.lick_coverage)
    params = GAParams(pop_size=args.pop, generations=args.gens, crossover_rate=args.crossover,
                      mutation_rate=args.mutation, tournament_k=args.tournament,
                      elitism=args.elitism, immigrants=args.immigrants,
                      patience=args.patience, init=args.init,
                      seed=args.seed, grid=_parse_grid(args.fastest),
                      lick_rate=args.lick_rate, fitness=fp)

    def progress(gen, hist, _best):
        if gen % 50 == 0:
            print(f"  geração {gen:4d}  melhor {hist['best'][-1]:.4f}  média {hist['mean'][-1]:.4f}")

    print(f"Progressão: {cfg.progression}  ({cfg.rhythm}, {cfg.bpm} bpm, modo {cfg.mode})")
    print(f"Escala do solo: {cfg.tonic} {SCALE_LABELS[cfg.scale]} "
          f"({' '.join(scale_note_names(cfg.tonic, cfg.scale))})")
    res = run_ga(cfg, params, progress)
    sp = res.space
    print(f"Figura mais rápida: {GRID_NAMES[sp.subdiv]} ({sp.subdiv}/tempo, "
          f"{cfg.bpm / 60 * sp.subdiv:.1f} notas/s); densidade alvo "
          f"{res.fitness_params.density} notas/tempo, shred alvo {res.fitness_params.shred}")
    print(f"Fim: {res.generations_run} gerações ({res.stop_reason}), {res.elapsed_sec:.1f} s, "
          f"fitness {res.best_fitness:.4f}")
    for k, v in res.breakdown.items():
        print(f"  {k:16s} {v:.3f}")
    licks = [u for u in lick_usage(res.best, sp) if u["intact"] >= 0.5]
    print("Licks no solo final (>= 50% intactos):")
    for u in licks:
        print(f"  tempo {u['beat']:5.1f}  {u['lick']} ({u['genre']}), {u['intact']:.0%}")

    name = args.name or f"{_default_name(args, cfg)}_{cfg.scale}_seed{args.seed}"
    out = Path(args.out) / name
    midis = build_song_midis(cfg, res.best, res.space, expressive=not args.no_expressive,
                             seed=args.seed, solo_db=args.solo_db)
    paths = write_midis(midis, out)
    plot_fitness({f"seed {args.seed}": res.history}, out / "fitness.png",
                 title=f"Evolução do fitness - {cfg.preset.label}")
    run = {
        "song": {"progression": cfg.progression, "rhythm": cfg.rhythm, "bpm": cfg.bpm,
                 "tonic": cfg.tonic, "mode": cfg.mode, "choruses": cfg.choruses,
                 "swing": cfg.swing, "solo_scale": cfg.scale},
        "ga": params.to_dict(),
        "result": {"fitness": res.best_fitness, "breakdown": res.breakdown,
                   "generations": res.generations_run, "stop_reason": res.stop_reason,
                   "elapsed_sec": res.elapsed_sec, "grid_subdiv": sp.subdiv,
                   "density_target": res.fitness_params.density,
                   "shred_target": res.fitness_params.shred,
                   "licks": licks},
        "genome": res.best.tolist(),
        "history": res.history,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    for k, p in paths.items():
        print(f"  MIDI {k:9s} -> {p}")
    print(f"  PNG  fitness   -> {out / 'fitness.png'}")
    if not args.no_audio:
        print(f"  WAV  completo  -> {render_wav(midis['completo'], out / 'completo.wav')}")


def _add_song_args(p):
    p.add_argument("--chords", help='progressão, um token por compasso (ex.: "A7 D7 A7 A7")')
    p.add_argument("--preset", default="12-bar blues", choices=list(PRESET_PROGRESSIONS))
    p.add_argument("--tonic", default="A")
    p.add_argument("--mode", choices=MODES, help="padrão: o modo da progressão pronta")
    p.add_argument("--rhythm", default="shuffle", choices=list(PRESETS))
    p.add_argument("--bpm", type=int)
    p.add_argument("--swing", type=float, help="0.5 = reto, 0.667 = shuffle")
    p.add_argument("--choruses", type=int, default=2)
    p.add_argument("--name")
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-audio", action="store_true")


def main():
    ap = argparse.ArgumentParser(description="Gerador de solos de guitarra (GA)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="lista ritmos, progressões e critérios da fitness")
    lp.add_argument("--tonic", default="A")
    lp.set_defaults(func=cmd_list)

    b = sub.add_parser("backing", help="gera só base + bateria (etapa 1)")
    _add_song_args(b)
    b.set_defaults(func=cmd_backing)

    s = sub.add_parser("solo", help="gera o solo por GA + base + bateria")
    _add_song_args(s)
    g = s.add_argument_group("algoritmo genético")
    d = GAParams()
    g.add_argument("--pop", type=int, default=d.pop_size)
    g.add_argument("--gens", type=int, default=d.generations)
    g.add_argument("--crossover", type=float, default=d.crossover_rate)
    g.add_argument("--mutation", type=float, default=d.mutation_rate)
    g.add_argument("--tournament", type=int, default=d.tournament_k)
    g.add_argument("--elitism", type=int, default=d.elitism)
    g.add_argument("--immigrants", type=float, default=d.immigrants,
                   help="fração da população trocada por indivíduos aleatórios a cada geração")
    g.add_argument("--patience", type=int, default=d.patience)
    g.add_argument("--init", choices=["walk", "uniform"], default=d.init)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--scale", default="auto", choices=["auto"] + list(SCALES),
                   help="escala do solo; auto = padrão do ritmo e do modo")
    g.add_argument("--density", type=float,
                   help="notas por tempo (média); padrão: perfil do ritmo")
    g.add_argument("--shred", type=float,
                   help="fração alvo de notas em rajadas rápidas; padrão: perfil do ritmo")
    g.add_argument("--fastest", default="auto",
                   help="figura mais rápida do solo: auto, " + ", ".join(
                       n.replace(" ", "_") for n in GRID_NAMES.values()))
    g.add_argument("--rest-ratio", type=float, default=d.fitness.rest_ratio,
                   help="fração alvo de tempo em pausa")
    g.add_argument("--weight", action="append", metavar="CRITÉRIO=PESO",
                   help="altera o peso de um critério (repetível); ver `cli.py list`")
    g.add_argument("--lick-rate", type=float, default=d.lick_rate,
                   help="chance de lick por compasso na população inicial (0 = sem licks)")
    g.add_argument("--lick-coverage", type=float, default=d.fitness.lick_coverage,
                   help="fração do solo em licks do estilo a partir da qual o idioma satura")
    g.add_argument("--no-expressive", action="store_true", help="sem bends/vibrato")
    g.add_argument("--solo-db", type=float, default=7.0,
                   help="quantos dB o solo fica acima do acompanhamento na mixagem")
    s.set_defaults(func=cmd_solo)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
