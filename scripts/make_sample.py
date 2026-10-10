"""Generate a synthetic baseline sample with native Windows TTS (SAPI).

Known transcript -> known WAV. No licensing worries (machine voice, we own it),
and the transcript is exact ground truth for word-level work later.

Usage:
    python scripts/make_sample.py            # writes data/raw/baseline_001.{wav,txt}
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Apostrophe-free so the PowerShell here-string stays simple.
TRANSCRIPT = (
    "Good evening everyone. "
    "Today I want to talk about the power of clear and confident communication. "
    "A great speech is not about big words. "
    "It is about rhythm, pauses, and the energy you bring to every sentence. "
    "Thank you for listening."
)


def synth_sapi(text: str, out_wav: pathlib.Path) -> None:
    """Speak `text` to `out_wav` via System.Speech (Windows only)."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate = 0;"
        f"$s.SetOutputToWaveFile('{out_wav}');"
        f"$s.Speak(@'\n{text}\n'@);"
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        check=True,
    )


def main() -> int:
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "baseline_001.wav"
    txt = out_dir / "baseline_001.txt"
    synth_sapi(TRANSCRIPT, wav)
    txt.write_text(TRANSCRIPT, encoding="utf-8")
    print(f"wrote {wav}")
    print(f"wrote {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
