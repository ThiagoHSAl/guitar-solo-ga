"""Interface Streamlit. Só UI: toda a lógica está no pacote solo_ga.

Executar:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import io
import tempfile
from pathlib import Path

import streamlit as st

from solo_ga.config import SongConfig
from solo_ga.ga import (CRITERIA, CRITERIA_LABELS, FitnessParams, GAParams, lick_usage,
                        run_ga)
from solo_ga.midi_io import build_backing_midis, build_song_midis
from solo_ga.plots import plot_fitness
from solo_ga.render import mp3_bytes, synthesize, wav_bytes
from solo_ga.ga.genome import GRID_NAMES, PLAYABLE_NPS, auto_subdiv, grid_options
from solo_ga.rhythms import (GENRES, PRESETS, SOLO_SPEED, default_solo_scale,
                             presets_by_genre)
from solo_ga.theory import (MODES, PRESET_PROGRESSIONS, SCALE_LABELS, SCALES,
                            scale_note_names, harmonic_field,
                            preset_progression)

st.set_page_config(page_title="Gerador de solos de guitarra", layout="wide")

MODE_LABELS = {"major": "Maior", "minor": "Menor", "blues": "Blues"}
TONICS = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B"]

ss = st.session_state
ss.setdefault("prog", preset_progression("12-bar blues", "A"))
ss.setdefault("genre", "Blues")
ss.setdefault("rhythm", "shuffle")
ss.setdefault("bpm", PRESETS["shuffle"].default_bpm)
ss.setdefault("approved", None)
ss.setdefault("mode", "blues")
ss.setdefault("tonic", "A")


# ------------------------------------------------------------------ callbacks

def _append_chord(sym: str):
    ss.prog = (ss.prog + " " + sym).strip()


def _undo_chord():
    ss.prog = " ".join(ss.prog.split()[:-1])


def _clear_prog():
    ss.prog = ""


def _load_preset():
    ss.prog = preset_progression(ss.preset_name, ss.tonic)
    ss.mode = PRESET_PROGRESSIONS[ss.preset_name]["mode"]


def _rhythm_changed():
    ss.bpm = PRESETS[ss.rhythm].default_bpm
    ss.swing_override = False


def _genre_changed():
    ss.rhythm = presets_by_genre(ss.genre)[0]
    _rhythm_changed()


# ------------------------------------------------------------------ etapa 1

st.title("Gerador de solos de guitarra")
st.caption("Algoritmo genético sobre uma progressão de acordes. "
           "Etapa 1: monte a base e ouça antes de gerar o solo.")

st.header("1. Harmonia e groove")
col_h, col_r = st.columns([3, 2], gap="large")

with col_h:
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.selectbox("Tônica", TONICS, key="tonic")
    c2.radio("Modo", MODES, format_func=MODE_LABELS.get, key="mode", horizontal=True)
    c3.toggle("Tétrades", key="sevenths", value=True)

    st.markdown("**Campo harmônico** (clique para adicionar um compasso)")
    field = harmonic_field(ss.tonic, ss.mode, ss.sevenths)
    btn_cols = st.columns(len(field))
    for col, (roman, sym) in zip(btn_cols, field):
        col.button(f"{sym}\n\n{roman}", key=f"add_{roman}_{sym}", on_click=_append_chord,
                   args=(sym,), width="stretch")

    p1, p2 = st.columns([3, 1], vertical_alignment="bottom")
    p1.selectbox("Progressões prontas", list(PRESET_PROGRESSIONS), key="preset_name")
    p2.button("Carregar", on_click=_load_preset, width="stretch")

    st.text_area("Progressão (um acorde por compasso; use vírgula para dois acordes no "
                 "mesmo compasso, ex.: Dm7,G7)", key="prog", height=90)
    u1, u2, _ = st.columns([1, 1, 3])
    u1.button("Desfazer último", on_click=_undo_chord, width="stretch")
    u2.button("Limpar", on_click=_clear_prog, width="stretch")

with col_r:
    g1, g2 = st.columns(2)
    g1.selectbox("Gênero", GENRES, key="genre", on_change=_genre_changed)
    g2.selectbox("Ritmo", presets_by_genre(ss.genre), key="rhythm", on_change=_rhythm_changed,
                 format_func=lambda k: PRESETS[k].label)
    preset = PRESETS[ss.rhythm]
    st.caption(preset.description)
    st.slider("BPM", 40, 200, key="bpm")
    st.number_input("Repetições da progressão (choruses)", 1, 6, 2, key="choruses")
    if st.toggle("Ajustar swing manualmente", key="swing_override"):
        swing = st.slider("Swing (0.50 = reto, 0.67 = shuffle)", 0.5, 0.75,
                          float(preset.swing), 0.01)
    else:
        swing = None

# Validação da progressão
try:
    cfg = SongConfig(ss.prog, rhythm=ss.rhythm, bpm=ss.bpm, tonic=ss.tonic, mode=ss.mode,
                     choruses=int(ss.choruses), swing=swing)
except ValueError as e:
    st.error(str(e))
    st.stop()

n_sec = (cfg.n_bars + 1) * 4 * cfg.beat_sec
st.info(f"{len(cfg.bars)} compassos x {cfg.choruses} = {cfg.n_bars} compassos "
        f"+ final, cerca de {n_sec:.0f} s a {cfg.bpm} bpm, swing {cfg.swing:.2f}")


@st.cache_data(show_spinner="Renderizando base e bateria...")
def render_backing(prog, rhythm, bpm, tonic, mode, choruses, swing):
    cfg = SongConfig(prog, rhythm=rhythm, bpm=bpm, tonic=tonic, mode=mode,
                     choruses=choruses, swing=swing)
    pm = build_backing_midis(cfg)["base_bateria"]
    audio = synthesize(pm)
    buf = io.BytesIO()
    pm.write(buf)
    return {"wav": wav_bytes(audio), "mp3": mp3_bytes(audio), "mid": buf.getvalue()}


def audio_downloads(name: str, files: dict[str, bytes], key: str):
    """Botões de download de `name` em WAV, MP3 e MIDI."""
    mimes = {"wav": "audio/wav", "mp3": "audio/mpeg", "mid": "audio/midi"}
    for col, ext in zip(st.columns(3), ["wav", "mp3", "mid"]):
        col.download_button(f"{name}.{ext}", files[ext], f"{name}.{ext}", mimes[ext],
                            type="primary" if ext == "wav" else "secondary",
                            width="stretch", key=f"dl_{key}_{ext}")


key = (ss.prog, ss.rhythm, int(ss.bpm), ss.tonic, ss.mode, int(ss.choruses), swing)
b1, b2, _ = st.columns([1, 1, 3])
listen = b1.button("Ouvir base", type="primary", width="stretch")

if listen or ss.get("last_key") == key:
    ss.last_key = key
    base = render_backing(*key)
    st.audio(base["wav"], format="audio/wav")
    audio_downloads("base", base, "base")
    if b2.button("Aprovar base", width="stretch"):
        ss.approved = key

# ------------------------------------------------------------------ etapa 2

def _midi_bytes(pm) -> bytes:
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


# Caráter do solo: multiplica o perfil de velocidade do ritmo (densidade, shred).
CHARACTERS = {"Melódico": (0.65, 0.3), "Equilibrado": (1.0, 1.0), "Virtuoso": (1.35, 1.8)}


def character_targets(cfg: SongConfig, character: str, sub: int) -> tuple[float, float]:
    """Densidade (notas por tempo) e shred para o caráter escolhido, limitados ao que a
    figura mais rápida permite: sem semicolcheias (ou sextinas) não há rajadas, e a
    densidade não passa de ~80% dos ataques possíveis."""
    dens0, shred0 = SOLO_SPEED[cfg.rhythm]
    kd, ks = CHARACTERS[character]
    density = min(dens0 * kd, 0.8 * sub)
    shred = min(0.8, shred0 * ks + (0.1 if character == "Virtuoso" else 0.0))
    if sub < 4:
        shred = 0.0
    return round(density, 2), round(shred, 2)


def _solo_panel(cfg: SongConfig):
    auto = default_solo_scale(cfg.rhythm, cfg.mode)
    sc1, sc2 = st.columns([1, 2], vertical_alignment="bottom")
    choice = sc1.selectbox(
        "Escala do solo", ["auto"] + list(SCALES), key="solo_scale",
        format_func=lambda k: (f"Automática: {SCALE_LABELS[auto]}" if k == "auto"
                               else SCALE_LABELS[k]))
    cfg = dataclasses.replace(cfg, solo_scale=choice)
    sc2.caption(f"{cfg.tonic} {SCALE_LABELS[cfg.scale]}: "
                f"{' '.join(scale_note_names(cfg.tonic, cfg.scale))}. A automática segue o "
                f"ritmo ({cfg.preset.label}) e o modo da progressão.")

    # As chaves incluem o ritmo: ao trocar de ritmo, os controles voltam ao perfil dele.
    auto_g = auto_subdiv(cfg)
    v1, v2, v3 = st.columns(3)
    grid = v1.selectbox(
        "Figura mais rápida", ["auto", *grid_options(cfg)], key=f"grid_{cfg.rhythm}",
        format_func=lambda g: (f"Automática: {GRID_NAMES[auto_g]}" if g == "auto"
                               else GRID_NAMES[g].capitalize()),
        help="Menor duração possível de nota no solo. A automática escolhe a mais fina "
             "que fique tocável no andamento da música.")
    sub = auto_g if grid == "auto" else grid
    character = v2.selectbox("Caráter do solo", list(CHARACTERS), index=1,
                             key=f"char_{cfg.rhythm}",
                             help="Ajusta juntos a densidade de notas e a quantidade de "
                                  "rajadas rápidas, a partir do perfil do ritmo.")
    density, shred = character_targets(cfg, character, sub)
    lick_rate = v3.slider("Frequência de licks", 0.0, 1.0, 0.3, 0.05,
                          help="Chance de cada compasso receber um lick do estilo. "
                               "0 = solo só por regras, sem licks.")
    nps = cfg.bpm / 60 * sub
    msg = (f"Figura mais rápida: {GRID_NAMES[sub]} ({nps:.1f} notas/s a {cfg.bpm} bpm). "
           f"Caráter {character.lower()}: {density:.1f} notas por tempo, "
           f"{shred:.0%} das notas em rajadas.")
    if nps > PLAYABLE_NPS:
        st.warning(msg + " A figura está acima do que um guitarrista toca com clareza.")
    else:
        st.caption(msg)

    w1, w2, w3 = st.columns(3)
    seed = w1.number_input("Semente", 0, 10_000, 42,
                           help="Mesma configuração com outra semente = outro solo.")
    rest = w2.slider("Pausa (fração do tempo)", 0.0, 0.6, 0.12, 0.02,
                     help="Quanto do solo é silêncio entre as frases.")
    expressive = w3.toggle("Bends e vibrato", value=True)

    with st.expander("Avançado"):
        a1, a2, a3, a4 = st.columns(4)
        pop = a1.number_input("População", 20, 500, 120, 10)
        gens = a2.number_input("Gerações (máx.)", 10, 2000, 300, 10)
        patience = a3.number_input("Parada por estagnação (gerações)", 10, 2000, 80, 10)
        init = a4.selectbox("População inicial", ["walk", "uniform"],
                            format_func={"walk": "Passeio na escala",
                                         "uniform": "Uniforme"}.get)
        b1, b2, b3, b4 = st.columns(4)
        mutation = b1.slider("Taxa de mutação", 0.0, 1.0, 0.3, 0.05)
        crossover = b2.slider("Taxa de cruzamento", 0.0, 1.0, 0.9, 0.05)
        tour = b3.slider("Tamanho do torneio", 2, 10, 3)
        elitism = b4.slider("Elitismo", 0, 10, 2)
        c1, c2, c3 = st.columns(3)
        immigrants = c3.slider("Imigrantes por geração (fração)", 0.0, 0.3, 0.05, 0.01,
                               help="Indivíduos novos que entram a cada geração para "
                                    "manter a diversidade da população.")
        lick_cov = c1.slider("Cobertura de licks alvo (fração do solo)", 0.0, 0.8, 0.35,
                             0.05)
        solo_db = c2.slider("Solo acima da base (dB)", 0.0, 15.0, 7.0, 0.5)

    with st.expander("Pesos da fitness"):
        wcols = st.columns(3)
        weights = {k: wcols[i % 3].slider(lbl, 0.0, 5.0, w, 0.5, key=f"w_{k}")
                   for i, (k, lbl, w) in enumerate(CRITERIA)}
    go = st.button("Gerar solo", type="primary")

    if go:
        params = GAParams(pop_size=int(pop), generations=int(gens), crossover_rate=crossover,
                          mutation_rate=mutation, tournament_k=tour, elitism=elitism,
                          immigrants=immigrants,
                          patience=int(patience), init=init, seed=int(seed),
                          grid=None if grid == "auto" else grid, lick_rate=lick_rate,
                          fitness=FitnessParams(weights=weights, density=density,
                                                shred=shred, rest_ratio=rest,
                                                lick_coverage=lick_cov))
        status = st.empty()
        chart = st.empty()

        def progress(gen, hist, _best):
            status.write(f"Geração {gen} - melhor {hist['best'][-1]:.4f}, "
                         f"média {hist['mean'][-1]:.4f}")
            chart.line_chart({"melhor": hist["best"], "média": hist["mean"],
                              "pior": hist["worst"]}, height=260)

        res = run_ga(cfg, params, progress)
        status.empty()
        chart.empty()
        midis = build_song_midis(cfg, res.best, res.space, expressive=expressive,
                                 seed=int(seed), solo_db=solo_db)
        with st.spinner("Renderizando áudio..."):
            audio = {k: synthesize(midis[k]) for k in ("completo", "solo")}
        with tempfile.TemporaryDirectory() as td:
            png = plot_fitness({f"seed {seed}": res.history}, Path(td) / "f.png",
                               title=f"Evolução do fitness - {cfg.preset.label}").read_bytes()
        ss.solo = {"base_key": ss.approved, "seed": int(seed), "res": res, "scale": cfg.scale,
                   "licks": [u for u in lick_usage(res.best, res.space) if u["intact"] >= 0.5],
                   "files": {k: {"wav": wav_bytes(a), "mp3": mp3_bytes(a),
                                 "mid": _midi_bytes(midis[k])} for k, a in audio.items()},
                   "png": png}

    sol = ss.get("solo")
    if not sol or sol["base_key"] != ss.approved:
        return
    res = sol["res"]
    st.markdown(f"**Resultado (semente {sol['seed']}, {SCALE_LABELS[sol['scale']]})**: "
                f"fitness {res.best_fitness:.4f} "
                f"após {res.generations_run} gerações ({res.stop_reason}, "
                f"{res.elapsed_sec:.1f} s)")
    r1, r2 = st.columns([3, 2], gap="large")
    with r1:
        st.line_chart({"melhor": res.history["best"], "média": res.history["mean"],
                       "pior": res.history["worst"]}, height=260)
        st.caption("Música completa")
        st.audio(sol["files"]["completo"]["wav"], format="audio/wav")
        st.caption("Só o solo")
        st.audio(sol["files"]["solo"]["wav"], format="audio/wav")
    with r2:
        st.dataframe({"critério": [CRITERIA_LABELS[k] for k in res.breakdown],
                      "nota": [round(v, 3) for v in res.breakdown.values()]},
                     hide_index=True, width="stretch")
        if sol["licks"]:
            st.markdown("**Licks no solo final**")
            st.dataframe({"tempo": [u["beat"] for u in sol["licks"]],
                          "lick": [u["lick"] for u in sol["licks"]],
                          "estilo": [u["genre"] for u in sol["licks"]],
                          "intacto": [f"{u['intact']:.0%}" for u in sol["licks"]]},
                         hide_index=True, width="stretch")
        else:
            st.caption("Nenhum lick sobreviveu inteiro à evolução.")
    st.caption("Música completa (base, bateria e solo)")
    audio_downloads("completo", sol["files"]["completo"], "completo")
    st.caption("Só o solo")
    audio_downloads("solo", sol["files"]["solo"], "solo")
    st.download_button("fitness.png", sol["png"], "fitness.png", "image/png")


st.header("2. Solo")
if ss.approved is None:
    st.write("Aprove uma base na etapa 1 para liberar a geração do solo.")
elif ss.approved != key:
    st.warning("A configuração mudou depois da aprovação. Ouça e aprove a base novamente.")
else:
    st.success("Base aprovada.")
    _solo_panel(cfg)
