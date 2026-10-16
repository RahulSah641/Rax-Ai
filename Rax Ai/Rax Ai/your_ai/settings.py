"""Small, atomic preferences file; document content lives in SQLite."""
import json
import os
from pathlib import Path
import tempfile

DEFAULTS = {
    "ollama_url": "http://127.0.0.1:11434",
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:1b",
    "samples_initialized": False,
}


def load_settings(directory):
    path = Path(directory) / "settings.json"
    result = DEFAULTS.copy()
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                for key, default in DEFAULTS.items():
                    if key in value and type(value[key]) is type(default):
                        result[key] = value[key]
        except (OSError, ValueError):
            pass  # Damaged preferences never prevent access to saved documents.
    return result


def save_settings(directory, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="settings-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({k: settings[k] for k in DEFAULTS}, stream, indent=2)
        os.replace(name, directory / "settings.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)
