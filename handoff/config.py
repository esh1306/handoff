"""Loads config.yaml + .env into a single dict-like config object."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """Dict that also allows attribute access, recursively."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def load_config(path=None):
    """Load config.yaml (defaulting to repo root) and .env, return a Config."""
    load_dotenv(REPO_ROOT / ".env")

    config_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    cfg = Config(data)
    cfg["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    cfg["elevenlabs_api_key"] = os.environ.get("ELEVENLABS_API_KEY", "")
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(f"fps={cfg.fps} window_s={cfg.window_s}")
    print(f"marker.dict={cfg.marker.dict} marker.id={cfg.marker.id}")
    print(f"OPENAI_API_KEY set: {bool(cfg.openai_api_key)}")
