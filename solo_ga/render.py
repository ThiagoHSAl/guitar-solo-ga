"""Renderização MIDI -> WAV/MP3 via FluidSynth."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import numpy as np
import pretty_midi

SAMPLE_RATE = 44100
ROOT = Path(__file__).resolve().parent.parent

# Ordem de busca do soundfont: variável de ambiente, pasta do projeto, soundfonts do sistema.
_CANDIDATES = [
    ROOT / "soundfonts" / "GeneralUser-GS.sf2",
    ROOT / "soundfonts" / "FluidR3_GM.sf2",
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
]


def find_soundfont() -> Path:
    env = os.environ.get("SOUNDFONT")
    if env:
        return Path(env)
    for p in list((ROOT / "soundfonts").glob("*.sf2")) + _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Nenhum soundfont encontrado. Baixe um .sf2 (ex.: GeneralUser GS) para soundfonts/ "
        "ou defina a variável SOUNDFONT."
    )


def synthesize(pm: pretty_midi.PrettyMIDI, sf2: str | Path | None = None,
               tail_sec: float = 2.0) -> np.ndarray:
    sf2 = str(sf2 or find_soundfont())
    audio = pm.fluidsynth(fs=SAMPLE_RATE, synthesizer=sf2)
    audio = np.concatenate([audio, np.zeros(int(tail_sec * SAMPLE_RATE))])
    peak = np.max(np.abs(audio)) or 1.0
    return audio / peak * 0.9


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def mp3_bytes(audio: np.ndarray, kbps: int = 192) -> bytes:
    """MP3 mono (LAME via `lameenc`), menor que o WAV para compartilhar."""
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(2)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return bytes(enc.encode(pcm.tobytes()) + enc.flush())


def render_wav(pm: pretty_midi.PrettyMIDI, path: str | Path, sf2: str | Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(synthesize(pm, sf2)))
    return path


# ------------------------------------------------------------------ mixagem automática

# Quanto o solo fica acima do acompanhamento (dB). Baixo e bateria ficam 1 e 2 dB acima
# do acompanhamento: sustentam o groove sem cobrir o solo.
SOLO_DB = 7.0


def mix_targets(solo_db: float = SOLO_DB) -> dict[str, float]:
    return {"solo": 0.0, "solo2": -3.0, "comp": -solo_db, "bass": -solo_db + 1,
            "drums": -solo_db + 2}
DEFAULT_CC7 = 100        # volume de canal padrão do General MIDI


def _role(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "drums"
    name = inst.name.lower()
    if "2ª voz" in name:
        return "solo2"
    if "solo" in name:
        return "solo"
    if "baixo" in name:
        return "bass"
    return "comp"


def _active_rms_db(audio: np.ndarray, floor_db: float = -50.0) -> float:
    """RMS só nas janelas de 50 ms em que a parte está soando (ignora pausas)."""
    w = int(0.05 * SAMPLE_RATE)
    n = len(audio) // w
    if n == 0:
        return -120.0
    frames = np.sqrt((audio[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-12)
    active = frames[frames > frames.max() * 10 ** (floor_db / 20)]
    return float(20 * np.log10(np.sqrt((active ** 2).mean()) + 1e-12))


def balance_mix(pm: pretty_midi.PrettyMIDI, sf2: str | Path | None = None,
                solo_db: float = SOLO_DB) -> dict[str, int]:
    """Mede cada instrumento isolado e grava um CC7 (volume de canal) que leva as partes
    aos níveis de `mix_targets(solo_db)`. Altera `pm` no lugar; devolve {nome: CC7}."""
    targets = mix_targets(solo_db)
    sf2 = str(sf2 or find_soundfont())
    gains = {}
    for inst in pm.instruments:
        if not inst.notes:
            continue
        inst.control_changes = [c for c in inst.control_changes if c.number != 7]
        level = _active_rms_db(inst.fluidsynth(fs=SAMPLE_RATE, synthesizer=sf2))
        gains[id(inst)] = targets[_role(inst)] - level
    if not gains:
        return {}
    # Ganho em dB -> CC7 (GM: amplitude ~ (cc/127)^2, ou seja, 40*log10 da razão).
    # Ancora o maior ganho em 127 para não saturar o volume de canal.
    top = max(gains.values())
    out = {}
    for inst in pm.instruments:
        if id(inst) not in gains:
            continue
        cc = int(round(127 * 10 ** ((gains[id(inst)] - top) / 40)))
        cc = max(1, min(127, cc))
        inst.control_changes.insert(0, pretty_midi.ControlChange(7, cc, 0.0))
        out[inst.name] = cc
    return out
