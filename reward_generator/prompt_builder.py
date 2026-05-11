from prompts.uav_navigation import SYSTEM_PROMPT, USER_TEMPLATE


def build_messages(task_name: str, prior_results: list = None) -> list[dict]:
    if task_name != "long_range_navigation":
        raise ValueError(f"Unsupported task: {task_name}")

    prior_results = prior_results or []
    feedback_block = _build_feedback_block(prior_results)
    user_content = USER_TEMPLATE
    if feedback_block:
        user_content += f"\n\n{feedback_block}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        lines += [
            "",
            ">>> RECENT FAILURES — DO NOT REPEAT THESE MISTAKES <<<",
        ]
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
                lines += [
                    "  Failing code snippet (first 10 lines):",
                    "  ---",
                    snippet,
                    "  ---",
                ]

    # ── Improvement directive ────────────────────────────────────
    directive = _build_improvement_directive(accepted, rejected, iteration)
    lines += [
        "",
        ">>> IMPROVEMENT DIRECTIVE FOR THIS ITERATION <<<",
        directive,
    ]

    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


def _build_improvement_directive(
    accepted: list, rejected: list, iteration: int
) -> str:
    directives = []

    # ── Metric-driven feedback ───────────────────────────────────
    if accepted:
        metrics = accepted[-1].get("runtime_test", {}).get("metrics", {})
        mean = metrics.get("mean")
        std  = metrics.get("std")

        if mean is not None:
            if mean < 0.05:
                directives.append(
                    f"Reward mean is very low ({mean:.3f}). Increase the "
                    f"distance_to_goal scale or add a progress bonus to give "
                    f"the agent a stronger learning signal."
                )
            elif mean > 0.8:
                directives.append(
                    f"Reward mean is very high ({mean:.3f}). This may cause "
                    f"training instability. Consider adding a penalty term or "
                    f"reducing scale constants."
                )

        if std is not None and std > 0.4:
            directives.append(
                f"Reward std is high ({std:.3f}) — reward varies wildly across "
                f"environments. Consider clamping components or normalizing scales."
            )

    # ── Iteration structural goals ───────────────────────────────
    iteration_goals = {
        0: "Produce a valid reward that passes all shape and validation checks.",
        1: "Improve reward density — every step should give a non-zero signal.",
        2: "Add a crash/out-of-bounds penalty: penalize Z < 0.1 or Z > 2.0.",
        3: "Add a progress bonus: reward the agent more when actively moving toward goal.",
        4: "Try potential-based shaping: reward = potential(current) - potential(next).",
    }
    goal = iteration_goals.get(iteration, "Improve reward quality and diversity.")
    directives.append(f"This iteration's goal: {goal}")

    # ── Diversity pressure after 2+ accepted ────────────────────
    if len(accepted) >= 2:
        directives.append(
            "IMPORTANT: Recent accepted rewards share similar structure. "
            "Try a meaningfully different approach — different components, "
            "different shaping functions, or different scale values."
        )

    # ── Repeated rejection pattern warning ──────────────────────
    if rejected:
        reasons = [r.get("reason") for r in rejected[-3:]]
        if reasons.count("runtime_test_failed") >= 2:
            directives.append(
                "WARNING: Multiple runtime failures detected. "
                "Double-check every component has shape (N,) before returning."
            )
        if reasons.count("too_similar_to_reference") >= 2:
            directives.append(
                "WARNING: Diversity check keeps failing. "
                "Introduce at least 3 new variable names not present in the reference."
            )

    return "\n".join(f"- {d}" for d in directives)


def _extract_message(record: dict) -> str:
    for key in ("runtime_test", "static_validation", "diversity_check"):
        block = record.get(key, {})
        if isinstance(block, dict) and block.get("message"):
            return block["message"]
    return "no message available"