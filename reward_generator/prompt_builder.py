"""
reward_generator/prompt_builder.py
Injects domain config values (batch_size, step_dt, tensor shapes) dynamically
so prompts/*.py never hardcode any numbers.
"""
import importlib
from pathlib import Path


def build_messages(domain, prior_results: list = None, failure_patch: str = "") -> list[dict]:
    # ── Load domain prompt module ─────────────────────────────────────────────
    try:
        mod = importlib.import_module(f"prompts.{domain.name}")
        system_prompt  = mod.SYSTEM_PROMPT
        tensor_reference = getattr(mod, "TENSOR_REFERENCE", "")
        few_shot       = getattr(mod, "FEW_SHOT_EXAMPLE", "")
        user_template  = mod.USER_TEMPLATE
    except ModuleNotFoundError:
        system_prompt  = "You are an expert robotics reward function engineer."
        tensor_reference = ""
        few_shot       = ""
        user_template  = (
            "Write a reward function for: {task_description}\n"
            "Baseline:\n```python\n{baseline_code}\n```\n"
            "{prior_results}\nOutput only a ```python ... ``` block."
        )

    # ── Load baseline code ────────────────────────────────────────────────────
    baseline_code = Path(domain.env_reference_path).read_text()

    # ── Build dynamic tensor reference from YAML tensor_shapes ───────────────
    dynamic_tensor_ref = _build_tensor_reference(domain)

    # ── Build dynamic constraints block from YAML ─────────────────────────────
    dynamic_constraints = _build_constraints(domain)

    # ── Format prior results (accepted previews + rejection reasons) ──────────
    prior_str = _format_prior_results(prior_results or [])

    # ── Pick variant ──────────────────────────────────────────────────────────
    n = len(prior_results) if prior_results else 0
    variant = ["from_scratch", "refine", "physics_guided"][n % 3]

    # ── Inject failure_patch into system prompt ───────────────────────────────
    full_system = system_prompt + "\n" + dynamic_constraints
    if failure_patch:
        full_system += "\n" + failure_patch

    # ── Fill user template ────────────────────────────────────────────────────
    user_content = user_template.format(
        tensor_reference = dynamic_tensor_ref,
        few_shot         = few_shot,
        baseline_code    = baseline_code,
        prior_results    = prior_str,
        variant          = variant,
        task_description = getattr(domain, "task_description", ""),
    )

    return [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_content},
    ]


def _build_tensor_reference(domain) -> str:
    tensor_shapes = getattr(domain, "tensor_shapes", {})
    cfg_scales    = getattr(domain, "cfg_scales", {})
    step_dt       = getattr(domain, "step_dt", 0.02)

    batch_size = 16
    for shape in tensor_shapes.values():
        if isinstance(shape, list) and len(shape) >= 1:
            batch_size = shape[0]
            break

    lines = [f"ALLOWED TENSORS (batch_size = N = {batch_size}):"]
    for name, shape in tensor_shapes.items():
        shape_str = f"[{', '.join(str(s) for s in shape)}]"
        lines.append(f"  self.{name:<45} shape {shape_str}")

    lines.append("")
    lines.append("ALLOWED CFG SCALES (float scalars):")
    for scale_name in cfg_scales:
        lines.append(f"  self.cfg.{scale_name}")

    lines.append("")
    lines.append(f"SCALAR: self.step_dt = {step_dt}")
    lines.append("")
    lines.append("NOTHING ELSE EXISTS. No self.num_envs, obs_dict, cfg[\"...\"], device.")

    return "\n".join(lines)


def _build_constraints(domain) -> str:
    tensor_shapes = getattr(domain, "tensor_shapes", {})
    cfg_scales    = getattr(domain, "cfg_scales", {})
    step_dt       = getattr(domain, "step_dt", 0.02)

    batch_size = 16
    for shape in tensor_shapes.values():
        if isinstance(shape, list) and len(shape) >= 1:
            batch_size = shape[0]
            break

    allowed_attrs = (
        [f"self.{n}" for n in tensor_shapes]
        + [f"self.cfg.{s}" for s in cfg_scales]
        + ["self.step_dt"]
    )

    lines = [
        "\nDOMAIN CONSTRAINTS (auto-generated from config):",
        f"  batch_size = {batch_size} (return tensor must be shape [{batch_size}])",
        f"  step_dt    = {step_dt}",
        f"  Allowed attributes ({len(allowed_attrs)} total):",
    ]
    for attr in allowed_attrs:
        lines.append(f"    {attr}")

    return "\n".join(lines)


def _format_prior_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["\nPREVIOUS ATTEMPTS THIS RUN:"]
    for r in results[-4:]:
        idx    = r.get("candidate_index", "?")
        status = r.get("status", "?")
        reason = r.get("reason", "")
        if status == "accepted":
            lines.append(f"  ✓ Candidate {idx}: ACCEPTED")
            code_preview = "\n".join((r.get("code") or "").splitlines()[:6])
            if code_preview:
                lines.append(f"```python\n{code_preview}\n```")
        else:
            rt_msg = r.get("runtime_test", {}).get("message", "")
            st_msg = r.get("static_validation", {}).get("message", "")
            lines.append(f"  ✗ Candidate {idx}: REJECTED — {reason}")
            # Include full error (stripped of verbose torch signatures)
            if rt_msg:
                cleaned = _clean_error(rt_msg)
                lines.append(f"    Runtime error: {cleaned}")
            if st_msg and st_msg != "OK":
                lines.append(f"    AST error: {st_msg}")
    return "\n".join(lines) + "\n"


def _clean_error(msg: str) -> str:
    """Strip verbose torch signature dumps, keep the meaningful first 2 lines."""
    import re
    cleaned = re.sub(r"expected one of:.*", "check torch docs for correct usage.", msg, flags=re.DOTALL)
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    return " | ".join(lines[:2])