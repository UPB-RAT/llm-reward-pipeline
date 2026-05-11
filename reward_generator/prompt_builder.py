import pathlib
import importlib


# Generic fallback SYSTEM_PROMPT — used when no domain-specific prompt file exists
DEFAULT_SYSTEM_PROMPT = """You are an expert reinforcement learning engineer.
Generate a Python method `_get_rewards(self) -> torch.Tensor` for an
IsaacLab DirectRLEnv subclass.

STRICT RULES:
- Function name must be exactly `_get_rewards`
- Only import/use torch (no numpy, no scipy, no custom modules)
- No forbidden calls: eval, exec, open, subprocess, os
- Output must be shape (N,) float32 tensor, no NaN, no Inf
- Use only self.robot.data.* attributes and self.cfg.* scales
- Return a single scalar reward per environment instance
"""


def _load_domain_prompts(domain) -> tuple[str, str | None]:
    """
    Try to load SYSTEM_PROMPT and USER_TEMPLATE from prompts/<domain.name>.py.
    Falls back to DEFAULT_SYSTEM_PROMPT + YAML task_description if not found.
    """
    try:
        mod = importlib.import_module(f"prompts.{domain.name}")
        system_prompt = getattr(mod, "SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        user_template = getattr(mod, "USER_TEMPLATE", None)
        return system_prompt, user_template
    except ModuleNotFoundError:
        return DEFAULT_SYSTEM_PROMPT, None


def build_messages(domain, prior_results: list = None) -> list[dict]:
    prior_results = prior_results or []

    system_prompt, user_template = _load_domain_prompts(domain)

    if user_template is not None:
        # Use the rich domain-specific USER_TEMPLATE (e.g. prompts/uav_navigation.py)
        user_content = user_template
    else:
        # Generic fallback — build from YAML task_description + reference env
        ref_code = pathlib.Path(domain.env_reference_path).read_text()
        user_content = f"""TASK DESCRIPTION:
{domain.task_description.strip()}

REFERENCE ENVIRONMENT (baseline — you MUST go beyond this):
```python
{ref_code}
```
"""

    feedback_block = _build_feedback_block(prior_results)
    if feedback_block:
        user_content += f"\n\n{feedback_block}"

    user_content += "\nGenerate ONLY a Python code block with the `_get_rewards` method."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


def _build_feedback_block(prior_results: list) -> str:
    if not prior_results:
        return ""

    accepted = [r for r in prior_results if r.get("status") == "accepted"]
    rejected = [r for r in prior_results if r.get("status") == "rejected"]
    iteration = len(prior_results)

    lines = ["=" * 60, "FEEDBACK FROM PREVIOUS CANDIDATES", "=" * 60]

    # ── Best accepted reward so far ──────────────────────────────
    if accepted:
        best = accepted[-1]
        metrics = best.get("runtime_test", {}).get("metrics", {})
        lines += [
            "",
            ">>> BEST ACCEPTED REWARD SO FAR <<<",
            "Build on this — keep what works, improve what doesn't.",
            "",
            best.get("code", "# (no code available)"),
            "",
            f"Runtime metrics: mean={metrics.get('mean', 'N/A'):.4f}  "
            f"std={metrics.get('std', 'N/A'):.4f}  "
            f"min={metrics.get('min', 'N/A'):.4f}  "
            f"max={metrics.get('max', 'N/A'):.4f}",
        ]
    else:
        lines += [
            "",
            ">>> NO ACCEPTED REWARD YET <<<",
            "No candidate has passed all checks. Try a different structure.",
        ]

    # ── Rejection signals ────────────────────────────────────────
    if rejected:
        recent_rejections = rejected[-3:]
        lines += ["", ">>> RECENT FAILURES — DO NOT REPEAT THESE MISTAKES <<<"]
        for r in recent_rejections:
            reason  = r.get("reason", "unknown")
            stage   = r.get("stage", "unknown")
            message = _extract_message(r)
            lines += [
                "",
                f"  Rejection stage  : {stage}",
                f"  Rejection reason : {reason}",
                f"  Validator message: {message}",
            ]
            code = r.get("code")
            if code and reason != "no_code_block":
                snippet = "\n".join(code.splitlines()[:10])
                lines += ["  Failing code snippet (first 10 lines):", "  ---", snippet, "  ---"]

    # ── Improvement directive ────────────────────────────────────
    directive = _build_improvement_directive(accepted, rejected, iteration)
    lines += ["", ">>> IMPROVEMENT DIRECTIVE FOR THIS ITERATION <<<", directive]
    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


def _build_improvement_directive(accepted: list, rejected: list, iteration: int) -> str:
    directives = []

    if accepted:
        metrics = accepted[-1].get("runtime_test", {}).get("metrics", {})
        mean = metrics.get("mean")
        std  = metrics.get("std")
        if mean is not None:
            if mean < 0.05:
                directives.append(
                    f"Reward mean is very low ({mean:.3f}). Increase the "
                    f"distance_to_goal scale or add a progress bonus."
                )
            elif mean > 0.8:
                directives.append(
                    f"Reward mean is very high ({mean:.3f}). Consider adding "
                    f"a penalty term or reducing scale constants."
                )
        if std is not None and std > 0.4:
            directives.append(
                f"Reward std is high ({std:.3f}). Consider clamping components "
                f"or normalizing scales."
            )

    iteration_goals = {
        0: "Produce a valid reward that passes all shape and validation checks.",
        1: "Improve reward density — every step should give a non-zero signal.",
        2: "Add a crash/out-of-bounds penalty: penalize Z < 0.1 or Z > 2.0.",
        3: "Add a progress bonus: reward the agent more when actively moving toward goal.",
        4: "Try potential-based shaping: reward = potential(current) - potential(next).",
    }
    directives.append(
        f"This iteration's goal: {iteration_goals.get(iteration, 'Improve reward quality and diversity.')}"
    )

    if len(accepted) >= 2:
        directives.append(
            "IMPORTANT: Recent accepted rewards share similar structure. "
            "Try a meaningfully different approach."
        )

    if rejected:
        reasons = [r.get("reason") for r in rejected[-3:]]
        if reasons.count("runtime_test_failed") >= 2:
            directives.append(
                "WARNING: Multiple runtime failures. "
                "Ensure every component has shape (N,) before returning."
            )
        if reasons.count("too_similar_to_reference") >= 2:
            directives.append(
                "WARNING: Diversity check keeps failing. "
                "Introduce at least 3 new variable names not in the reference."
            )

    return "\n".join(f"- {d}" for d in directives)


def _extract_message(record: dict) -> str:
    for key in ("runtime_test", "static_validation", "diversity_check"):
        block = record.get(key, {})
        if isinstance(block, dict) and block.get("message"):
            return block["message"]
    return "no message available"