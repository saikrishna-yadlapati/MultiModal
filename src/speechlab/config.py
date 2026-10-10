"""Config loader. Single source for paths, seeds, feature params, thresholds.

Loads config.yaml into a plain dict. Kept deliberately tiny — no settings framework
until one is actually needed (YAGNI).
"""
from __future__ import annotations

from pathlib import Path
import yaml

# Repo root = three levels up from this file (src/speechlab/config.py).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config into a dict. Raises if the file is missing."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Self-check: config loads and has the keys later sessions depend on.
    cfg = load_config()
    for key in ("seed", "paths", "audio", "features", "detection", "evaluation"):
        assert key in cfg, f"missing config key: {key}"
    assert cfg["audio"]["target_sample_rate"] > 0
    print("config OK:", DEFAULT_CONFIG_PATH)
