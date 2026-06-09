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
uav-reward-gen/
├── README.md
├── environment.yaml          # primary: conda setup
├── requirements.txt          # fallback: pip-only setup
├── configs/
│   └── default.yaml
├── prompts/
│   ├── __init__.py
│   └── uav_navigation.py
├── reward_generator/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── llm_client.py
│   ├── orchestrator.py
│   ├── prompt_builder.py
│   └── reward_store.py
├── validators/
│   ├── __init__.py
│   ├── ast_validator.py
│   ├── code_extractor.py
│   └── runtime_tester.py
├── utils/
│   ├── __init__.py
│   └── io.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
└── outputs/
    ├── rewards/
    └── logs/
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
git clone https://github.com/your-username/uav-reward-gen.git
cd uav-reward-gen
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

Run the pipeline to generate and validate reward function candidates:

```bash
python -m reward_generator.cli \
  --config configs/quadcopter_navigation.yaml \
  --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --num-candidates 5 \
  --clean
```

Results are saved to:
- `outputs/rewards/` — accepted reward `.py` files
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
[LLM: Qwen2.5-Coder-7B-Instruct (local)]
     │  generates reward function code
     ▼
[Code Extractor]  ← strips markdown fences
     │
     ▼
[AST Validator]   ← static safety checks
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