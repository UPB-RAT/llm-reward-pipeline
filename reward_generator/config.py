from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    temperature: float
    top_p: float
    max_tokens: int
    n_ctx: int
    n_gpu_layers: int


@dataclass
class PipelineConfig:
    num_candidates: int
    task_name: str
    save_all_raw: bool
    random_seed: int


@dataclass
class RuntimeTestConfig:
    batch_size: int
    max_episode_len: int


@dataclass
class AppConfig:
    model: ModelConfig
    pipeline: PipelineConfig
    runtime_test: RuntimeTestConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(
        model=ModelConfig(**data["model"]),
        pipeline=PipelineConfig(**data["pipeline"]),
        runtime_test=RuntimeTestConfig(**data["runtime_test"]),
    )