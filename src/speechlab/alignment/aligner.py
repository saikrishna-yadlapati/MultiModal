"""Forced alignment via torchaudio MMS_FA (Wav2Vec2 CTC) — Session 2 decision.

Chosen over aeneas/WhisperX/MFA because it is pip-only, CPU-capable, has Python 3.13
wheels, and is the least fragile on Windows. It downloads one acoustic model on first
use (needs internet once; cached afterwards). Produces per-word [start,end] seconds.

Numbers come from the model's frame grid, not from any LLM. If the model cannot be
loaded (offline, no cache), callers should fall back to the energy-based word spread
(documented in docs/DECISIONS.md) rather than fabricate timestamps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_ALIGN_SR = 16000  # MMS_FA operates at 16 kHz; we resample to match.


@dataclass
class WordSpan:
    word: str          # normalized word actually aligned
    start_s: float
    end_s: float
    score: float       # mean per-frame alignment confidence, 0..1
    orig_index: int    # index into the normalized word list


def _normalize(transcript: str, vocab: set[str]) -> list[str]:
    """Lowercase, keep only characters the model knows, split into words.

    Words with no in-vocabulary character are dropped (can't be aligned).
    """
    words = []
    for raw in re.split(r"\s+", transcript.lower().strip()):
        kept = "".join(ch for ch in raw if ch in vocab)
        if kept:
            words.append(kept)
    return words


def align_words(y: np.ndarray, sr: int, transcript: str) -> list[WordSpan]:
    """Force-align `transcript` to audio `y`. Returns per-word spans in seconds.

    Raises RuntimeError if the model/bundle can't be obtained — caller decides fallback.
    """
    import torch
    import torchaudio

    try:
        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        vocab = set(bundle.get_dict().keys())
    except Exception as e:  # offline / download failure
        raise RuntimeError(f"MMS_FA unavailable (likely offline): {e}") from e

    words = _normalize(transcript, vocab)
    if not words:
        raise RuntimeError("no alignable words after normalization")

    wav = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)  # [1, time]
    if sr != _ALIGN_SR:
        wav = torchaudio.functional.resample(wav, sr, _ALIGN_SR)

    with torch.inference_mode():
        emission, _ = model(wav)
        token_spans = aligner(emission[0], tokenizer(words))

    num_frames = emission.size(1)
    ratio = wav.size(1) / num_frames          # audio samples per emission frame
    to_s = lambda frame: float(frame) * ratio / _ALIGN_SR  # noqa: E731

    out: list[WordSpan] = []
    for i, (word, spans) in enumerate(zip(words, token_spans)):
        total = sum(s.end - s.start for s in spans) or 1
        score = sum(s.score * (s.end - s.start) for s in spans) / total
        out.append(WordSpan(word, to_s(spans[0].start), to_s(spans[-1].end),
                            float(score), i))
    return out
