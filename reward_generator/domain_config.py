from dataclasses import dataclass, field
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


# ── NEW: domain adapter fields ────────────────────────────
@dataclass
class DomainConfig:
    name: str                        # e.g. "uav_navigation"
    task_description: str            # injected into LLM prompt
    env_reference_path: str          # path to reference _get_rewards
    step_dt: float                   # used in runtime tester
    tensor_shapes: dict              # { "robot.data.root_lin_vel_b": [16, 3] }
    cfg_scales: dict                 # { "lin_vel_reward_scale": -0.05 }


@dataclass
class AppConfig:
    model: ModelConfig
    pipeline: PipelineConfig
    runtime_test: RuntimeTestConfig
    domain: DomainConfig             # ← NEW


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(
        model=ModelConfig(**data["model"]),
        pipeline=PipelineConfig(**data["pipeline"]),
        runtime_test=RuntimeTestConfig(**data["runtime_test"]),
        domain=DomainConfig(**data["domain"]),   # ← NEW
    )