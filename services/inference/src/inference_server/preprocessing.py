"""Turn an uploaded audio file into fixed-length clips for the model.

Pipeline: decode → mono → resample → split into `clip_seconds` windows (last one zero-padded).
"""

import logging
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


class AudioError(ValueError):
    """The file can't be used as model input: missing, unreadable, or empty."""


def load_clips(
    path: str | Path,
    sample_rate: int = 16_000,
    clip_seconds: float = 4.0,
    min_tail_seconds: float = 1.0,
    upload_dir: str | Path | None = None,
) -> list[np.ndarray]:
    """Load an audio file as a list of mono float32 clips of exactly `clip_seconds`.

    If `upload_dir` is given, files outside it are rejected without being opened.
    """
    if upload_dir is not None and not Path(path).resolve().is_relative_to(
        Path(upload_dir).resolve()
    ):
        raise AudioError(f"Path {path} is outside the uploads directory")

    waveform = load_waveform(path, sample_rate)
    return split_into_clips(
        waveform,
        clip_samples=int(sample_rate * clip_seconds),
        min_tail_samples=int(sample_rate * min_tail_seconds),
    )


def load_waveform(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode the file into a mono float32 waveform at `sample_rate`."""
    try:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as e:  # sf.LibsndfileError is a RuntimeError
        raise AudioError(f"Cannot read audio file {Path(path).name}: {e}") from e

    if audio.size == 0:
        raise AudioError(f"Audio file {Path(path).name} is empty")

    logger.debug(
        "Decoded %s: %.2f s, %d Hz, %d channel(s)",
        Path(path).name,
        audio.shape[0] / source_rate,
        source_rate,
        audio.shape[1],
    )

    waveform = audio.mean(axis=1)  # (frames, channels) → mono

    if source_rate != sample_rate:
        g = gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // g, source_rate // g)

    return waveform.astype(np.float32, copy=False)


def split_into_clips(
    waveform: np.ndarray, clip_samples: int, min_tail_samples: int
) -> list[np.ndarray]:
    """Cut the waveform into consecutive `clip_samples` windows and zero-pad the last one.

    A trailing piece shorter than `min_tail_samples` is dropped, since it would be mostly
    padding and give an unreliable vote. Files shorter than one clip are always kept and padded.
    """
    clips = [waveform[i : i + clip_samples] for i in range(0, len(waveform), clip_samples)]

    if len(clips) > 1 and len(clips[-1]) < min_tail_samples:
        clips.pop()

    if len(clips[-1]) < clip_samples:
        clips[-1] = np.pad(clips[-1], (0, clip_samples - len(clips[-1])))

    return clips
