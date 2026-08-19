# UAV Reward Generation Pipeline

A local, open-source LLM pipeline for generating and validating reward functions for UAV long-range navigation, designed to run fully offline on a machine with **RTX Ada 4000 12GB VRAM**.

This repository implements **Phase 1**:
1. Local LLM inference (Qwen2.5-Coder-7B-Instruct)
2. Prompt-based reward function synthesis
3. Code extraction, AST validation, and runtime smoke testing

**Phase 2** (IsaacLab integration and evolutionary training loop) will be built on top of this foundation.

---

## Repository Structure

```text
llm-reward-pipeline/
├── README.md
├── environment.yaml              # primary: conda setup
├── requirements.txt              # fallback: pip-only setup
├── configs/
│   ├── default.yaml              # UAV navigation defaults
│   └── quadcopter.yaml           # Quadcopter hover/nav domain config
├── prompts/
│   ├── __init__.py
│   ├── uav_navigation.py         # Prompts for long_range_navigation task
│   ├── quadcopter.py             # Prompts for quadcopter task
│   └── experimental.py           # Scratch prompt for testing prompt variants
├── reward_generator/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point (argparse)
│   ├── config.py                 # Dataclass config loader (YAML)
│   ├── llm_client.py             # Local GGUF client (llama-cpp-python)
│   ├── hf_client.py              # HuggingFace Transformers client (with LoRA)
│   ├── orchestrator.py           # Main generation loop + validation pipeline
│   ├── prompt_builder.py         # Dynamic prompt construction with feedback
│   ├── prompt_loader.py          # Load prompt lists from .py/.json/text files
│   └── reward_store.py           # Save accepted/rejected candidates to disk
├── validators/
│   ├── __init__.py
│   ├── ast_validator.py          # Static safety checks via Python AST
│   ├── code_extractor.py         # Strips markdown fences from LLM output
│   ├── diversity_checker.py      # Ensures novelty vs. reference components
│   └── runtime_tester.py         # Executes function on dummy tensors
├── envs/
│   └── quadcopter_env_reference.py  # Full IsaacLab QuadcopterEnv reference
├── utils/
│   ├── __init__.py
│   └── io.py                     # File I/O helpers
├── scripts/
│   └── rate_calc.py              # Typer CLI to compute accept/reject rates
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
└── outputs/
    ├── rewards/                  # Accepted reward .py + .json files
    └── logs/                     # Full generation logs
```

---

## Prerequisites

Before starting, make sure the following are available on your system.

Check NVIDIA driver and CUDA version:

```bash
nvidia-smi
```

Check CUDA compiler (required for building llama-cpp-python):

```bash
nvcc --version
```

If `nvcc` is not found, install the CUDA toolkit:

```bash
sudo apt install nvidia-cuda-toolkit
```

Check build tools (required for compiling llama-cpp-python):

```bash
gcc --version && cmake --version
```

If missing, install them:

```bash
sudo apt install build-essential cmake
```

Check conda is installed:

```bash
conda --version
```

If not installed, download Miniconda from https://docs.conda.io/en/latest/miniconda.html

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/llm-reward-pipeline.git
cd llm-reward-pipeline
```

---

## Step 2 — Create the Conda Environment

This creates a fully isolated Python 3.11 environment with PyTorch and CUDA 13.0 dependencies.
It will not affect any other environments or libraries on your machine.

```bash
conda env create -f environment.yaml
```

This installs:
- Python 3.11
- PyTorch >= 2.6 with CUDA 13.0 support
- NumPy, PyYAML, huggingface-hub, pytest, and other core dependencies

Verify the environment was created:

```bash
conda env list
```

You should see `uav-reward-gen` in the list.

---

## Step 3 — Activate the Environment

```bash
conda activate uav-reward-gen
```

> All subsequent commands must be run inside this activated environment.

---

## Step 4 — Build and Install llama-cpp-python with CUDA Support

No prebuilt CUDA 13.0 wheel is available yet for llama-cpp-python on Linux.
It must be compiled from source. This step takes approximately 5–10 minutes.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

This uses your system CUDA 13.0 toolkit automatically and compiles with full GPU support
for the RTX Ada 4000 (sm_89 architecture).

Verify the install:

```bash
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

---

## Step 5 — Verify PyTorch Detects the GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:

```
CUDA available: True
GPU: NVIDIA RTX 4000 Ada Generation
```

---

## Step 6 — Download the Model

Download the recommended GGUF quantized model into the `models/` directory:

```bash
mkdir -p models

huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir models/
```

> The Q4_K_M quantized model uses approximately 5–6 GB VRAM,
> leaving enough headroom on the 12 GB RTX Ada 4000 for development
> and later IsaacLab integration.

---

## Step 7 — Run the Tests

Verify the full pipeline (extraction, validation, smoke test) works correctly.
This step does not require the model to be loaded.

```bash
pytest tests/test_pipeline.py -v
```

Expected output:

```
tests/test_pipeline.py::test_extract_validate_runtime PASSED
```

---

## Step 8 — Generate Reward Candidates

### CLI Flags

| Flag | Required | Description |
|---|---|---|
| `--model-path` | Yes | GGUF file path or HuggingFace model ID |
| `--adapter-path` | No | LoRA adapter path (GGUF `.bin` or HF adapter ID) |
| `--client-type` | No | `local` (llama.cpp) or `hf` (Transformers). Auto-detected if not specified |
| `--num-candidates` | No | Number of reward functions to generate (default: config value) |
| `--feedback` | No | Enable the feedback loop: distill runtime failures from previous candidates into a token-free "failure scorecard" injected into the prompt |
| `--task` | No | Task name: `long_range_navigation` or `quadcopter` |
| `--prompt-style` | No | Prompt template style: `detailed` (per-domain/task prompt) or `experimental` (scratch prompt in `prompts/experimental.py`). Defaults to config value |
| `--prompts-file` | No | Path to a file with a list of prompts to run inference on (`.py` module exposing `TEST_PROMPTS`, `.json` array, or text file). When set, each candidate uses one prompt from the list |
| `--config` | No | Path to YAML config (default: `configs/default.yaml`) |
| `--clean` | No | Delete all previous outputs before running |

---

### Option A — Local GGUF Model (Offline)

Run the pipeline to generate and validate reward function candidates using local `.gguf` checkpoints (via llama-cpp-python):

**UAV Navigation task (`long_range_navigation`):**

```bash
python -m reward_generator.cli \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 5 \
  --task long_range_navigation
```

**Quadcopter task (`quadcopter`):**

```bash
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 5 \
  --task quadcopter
```

**With a local GGUF LoRA adapter:**

```bash
python -m reward_generator.cli \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --adapter-path models/your-adapter.bin \
  --num-candidates 5 \
  --task quadcopter
```

### Option B — HuggingFace Transformers Model (Base & Fine-tuned)

You can run base or fine-tuned model adapters directly from HuggingFace using the `HFLLMClient`. The backend client type is auto-detected as `hf` if `--model-path` is not a `.gguf` file.

**UAV Navigation task (`long_range_navigation`):**

```bash
# Base model
python -m reward_generator.cli \
  --model-path unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --num-candidates 5 \
  --task long_range_navigation

# Fine-tuned model (no adapter — full SFT weights)
python -m reward_generator.cli \
  --model-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 5 \
  --task long_range_navigation

# Base model + LoRA adapter
python -m reward_generator.cli \
  --model-path unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --adapter-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 5 \
  --task long_range_navigation
```

**Quadcopter task (`quadcopter`):**

Uses `configs/quadcopter.yaml` which includes domain-specific tensor shapes, cfg scales, and environment constraints:

```bash
# Base model
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --num-candidates 5 \
  --task quadcopter

# Fine-tuned model (full SFT weights)
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 5 \
  --task quadcopter

# Generate 1000 candidates, no feedback
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 1000 \
  --task quadcopter

# Generate with feedback enabled
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 50 \
  --feedback \
  --task quadcopter

# Clean previous outputs and regenerate
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path UPB-RAT-Lab/qwen2.5-coder-7b-sft-v2-huyen-889 \
  --num-candidates 10 \
  --clean \
  --task quadcopter
```

### Prompt Styles

The pipeline supports multiple prompt templates, selected via `--prompt-style` (or `prompt_style` in the YAML `pipeline` section):

- **`detailed`** (default) — the full per-domain prompt with tensor reference, few-shot example, and strict coding constraints (`prompts/quadcopter.py`, `prompts/uav_navigation.py`).
- **`experimental`** — a bare scratch prompt from `prompts/experimental.py`. Edit that file to test out different prompts and inferences — no code changes required. By default the prompt is sent as-is; pass `--feedback` to append the failure scorecard.

### How feedback works

Feedback has two channels, both derived from the previous candidates in the run. Both are **token-free by design**: the model is never told the name of an attribute it hallucinated, because repeating a plausible-but-wrong token back into the prompt anchors the model and makes it repeat the mistake (this measurably dropped acceptance when attribute names were fed back verbatim).

- **Channel A — failure scorecard (into the system prompt, `--feedback` only).** Runtime-test failures are tallied into error *categories* — `missing_attr`, `dim_out_of_range`, `shape_mismatch`, `invalid_output`, `output_shape`, `other_runtime` — each rendered as one sentence with a count (e.g. "attempts referenced a `self.*` attribute that does NOT exist in the sandbox (seen 2×)"). The patch then positively states the exact attribute surface the sandbox exposes (auto-derived from `validators/runtime_tester.py`), so the model knows what it *may* use instead of what it must not. Only candidates that reached the runtime smoke test and failed contribute.
- **Channel B — prior results window (into the user prompt, always rendered).** The last 4 candidates fill the `{prior_results}` placeholder of the domain template: accepted candidates show a 6-line code preview (a known-good structural template); rejected candidates show their reason plus a sanitized one-line *category label* — raw error messages are stripped of quoted tokens and dummy class names (`_Data`, `_DummyEnv`).

**Experimental style (`--prompt-style experimental`):**

```bash
# bare test prompt (user message only, exactly as written in prompts/experimental.py)
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 5 \
  --task quadcopter \
  --prompt-style experimental

# with the feedback loop enabled
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 5 \
  --task quadcopter \
  --prompt-style experimental \
  --feedback
```

> The selected `prompt_style` is recorded in each candidate's metadata/log output.

### Running Inference on a List of Prompts

To run inference on an external list of prompts (e.g. a 1,000-prompt test set), point the pipeline at the file with `--prompts-file`. Each candidate then uses one prompt from the list verbatim (as a user-only message); if `--num-candidates` exceeds the list length, the list wraps around.

Supported file formats:

- **`.py`** — a Python module exposing `TEST_PROMPTS` (or `PROMPTS` / `PROMPT_LIST`), a list of strings. Multi-line prompts supported.
- **`.json`** — a JSON array of strings. Multi-line prompts supported.
- **any other** — plain text, one prompt per non-empty line.

```bash
# Run one inference per prompt over a 1,000-prompt test set
python -m reward_generator.cli \
  --config configs/quadcopter.yaml \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 1000 \
  --task quadcopter \
  --prompts-file /path/to/test_prompts.py
```

Each candidate's metadata/log records `prompt_index` and the full `prompt` text, so you can trace which prompt produced which reward function. Feedback is not applied in this mode — every prompt is sent exactly as written.

Results are saved to:
- `outputs/rewards/` — accepted reward `.py` files + `.json` metadata
- `outputs/logs/` — full generation logs in `.json` format

---

## Environment Management

```bash
# Deactivate the environment when done
conda deactivate

# List all conda environments
conda env list

# Remove the environment completely (clean slate)
conda env remove -n uav-reward-gen

# Export a locked snapshot of the current environment
conda env export > environment.lock.yaml
```

---

## Alternative — pip only (fallback)

If you cannot use conda, install dependencies directly with pip:

```bash
pip install -r requirements.txt
```

Then build llama-cpp-python from source (same as Step 4):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

---

## Phase 1 Pipeline Overview

```
[Task Prompt]
     │
     ▼
[LLM: Qwen2.5-Coder-7B-Instruct (local or HuggingFace)]
     │  generates reward function code
     ▼
[Code Extractor]  ← strips markdown fences
     │
     ▼
[AST Validator]   ← static safety checks
     │
     ▼
[Diversity Checker] ← ensures ≥2 novel components vs. reference
     │
     ▼
[Runtime Tester]  ← smoke test on dummy tensors
     │
     ▼
[RewardStore]     ← saves accepted .py + metadata .json
```

---

## Phase 2 (Coming Next)

Phase 2 will extend this pipeline with:
- IsaacLab environment integration
- Per-candidate RL rollout evaluation
- Fitness scoring and ranking
- Evolutionary feedback loop back into the LLM prompt