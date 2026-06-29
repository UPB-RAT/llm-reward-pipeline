import argparse
import shutil
from pathlib import Path

from reward_generator.config import load_config
from reward_generator.llm_client import LocalLLMClient
from reward_generator.hf_client import HFLLMClient
from reward_generator.orchestrator import RewardGenerationOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="UAV reward generation pipeline")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--model-path",     required=True)
    parser.add_argument("--adapter-path",   default=None, help="Path/ID of the LoRA adapter (GGUF or HF).")
    parser.add_argument(
        "--client-type",
        choices=["local", "hf"],
        default=None,
        help="LLM client backend: 'local' (llama.cpp) or 'hf' (Transformers). Auto-detected if not specified.",
    )
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--task",           dest="task_name", default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all previous outputs (rewards/ and logs/) before running.",
    )
    return parser.parse_args()


def clean_outputs():
    for folder in ["outputs/rewards", "outputs/logs"]:
        p = Path(folder)
        if p.exists():
            shutil.rmtree(p)
            print(f"  Cleaned: {folder}/")
        p.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {folder}/")


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.num_candidates is not None:
        config.pipeline.num_candidates = args.num_candidates
    if args.task_name is not None:
        config.pipeline.task_name = args.task_name

    if args.clean:
        print("\n── Cleaning previous outputs ──")
        clean_outputs()

    client_type = args.client_type
    if client_type is None:
        client_type = "local" if args.model_path.endswith(".gguf") or Path(args.model_path).is_file() else "hf"

    if client_type == "local":
        llm = LocalLLMClient(
            base_model_path=args.model_path,
            adapter_path=args.adapter_path,
            n_ctx=config.model.n_ctx,
            n_gpu_layers=config.model.n_gpu_layers,
        )
    else:
        llm = HFLLMClient(
            base_model_path=args.model_path,
            adapter_path=args.adapter_path,
            n_ctx=config.model.n_ctx,
            n_gpu_layers=config.model.n_gpu_layers,
        )
    orchestrator = RewardGenerationOrchestrator(llm, config)
    accepted = orchestrator.run()

    print(f"Accepted reward functions saved: {len(accepted)}")
    for item in accepted:
        print(f"  → {item['code_path']}")


if __name__ == "__main__":
    main()