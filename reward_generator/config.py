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
    feedback: bool = False
    prompt_style: str = "detailed"
    prompts_file: str | None = None


@dataclass
class RuntimeTestConfig:
    batch_size: int
    max_episode_len: int


@dataclass
class DomainConfig:
    name: str
    task_description: str
    env_reference_path: str
    step_dt: float
    cfg_scales: dict[str, float]
    tensor_shapes: dict[str, list[int]]


@dataclass
class AppConfig:
    model: ModelConfig
    pipeline: PipelineConfig
    runtime_test: RuntimeTestConfig
    domain: DomainConfig | None = None


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    domain_data = data.get("domain")
    return AppConfig(
        model=ModelConfig(**data["model"]),
        pipeline=PipelineConfig(**data["pipeline"]),
        runtime_test=RuntimeTestConfig(**data["runtime_test"]),
        domain=DomainConfig(**domain_data) if domain_data else None,
    )