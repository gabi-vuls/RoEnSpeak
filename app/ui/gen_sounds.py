"""Generate the committed start/stop beep sounds.

    python -m app.ui.gen_sounds [--out DIR]     (default DIR: <repo>/assets/sounds)

start.wav: two short rising tones (660 Hz -> 990 Hz). stop.wav: the same, falling.
16-bit mono PCM, 44.1 kHz, ~180 ms each, soft attack/release so they never click.
"""

from __future__ import annotations

import argparse
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("roenspeak.ui.gen_sounds")

RATE = 44_100
VOLUME = 0.32


def _tone(freq: float, dur_s: float) -> np.ndarray:
    t = np.arange(int(RATE * dur_s)) / RATE
    # Fundamental plus a quiet octave for a softer, "chime"-like timbre.
    sig = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    attack = int(0.006 * RATE)
    env = np.exp(-t * 18.0)
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    release = int(0.012 * RATE)
    env[-release:] *= np.linspace(1.0, 0.0, release)
    return sig * env / 1.25


def _sequence(freqs: tuple[float, float]) -> np.ndarray:
    gap = np.zeros(int(0.02 * RATE))
    parts = [_tone(freqs[0], 0.08), gap, _tone(freqs[1], 0.08)]
    return np.concatenate(parts) * VOLUME


def _write(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16.tobytes())


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = out_dir / "start.wav"
    stop = out_dir / "stop.wav"
    _write(start, _sequence((660.0, 990.0)))
    _write(stop, _sequence((990.0, 660.0)))
    return [start, stop]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "assets" / "sounds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for p in generate(args.out):
        log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
