from pathlib import Path
from utils.io import ensure_dir, write_text, write_json


class RewardStore:
    def __init__(self, rewards_dir: str, logs_dir: str):
        self.rewards_dir = ensure_dir(rewards_dir)
        self.logs_dir = ensure_dir(logs_dir)

    def save_candidate(self, name: str, code: str, metadata: dict):
        code_path = Path(self.rewards_dir) / f"{name}.py"
        meta_path = Path(self.logs_dir) / f"{name}.json"
        write_text(code_path, code)
        write_json(meta_path, metadata)
        return str(code_path), str(meta_path)