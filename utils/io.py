from pathlib import Path
import json
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_text(path: str | Path, content: str):
    Path(path).write_text(content, encoding="utf-8")


def write_json(path: str | Path, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")