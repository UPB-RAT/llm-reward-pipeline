import argparse
import shutil
from pathlib import Path

from reward_generator.domain_config import load_config
from reward_generator.llm_client import LocalLLMClient
from reward_generator.orchestrator import RewardGenerationOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based reward generation pipeline")
    parser.add_argument("--config",         default="configs/default.yaml",
                        help="Path to YAML config (swap this to change domain)")
    parser.add_argument("--model-path",     required=True,
                        help="Path to GGUF model weights")
    parser.add_argument("--num-candidates", type=int, default=None,
                        help="Override pipeline.num_candidates from config")
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

    # CLI overrides
    if args.num_candidates is not None:
        config.pipeline.num_candidates = args.num_candidates

    if args.clean:
        print("\n── Cleaning previous outputs ──")
        clean_outputs()

    print(f"\n── Domain  : {config.domain.name}")
    print(f"── Config  : {args.config}")
    print(f"── Model   : {args.model_path}")
    print(f"── Candidates per run: {config.pipeline.num_candidates}\n")

    llm = LocalLLMClient(
        model_path=args.model_path,
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