import importlib


TASK_TO_PROMPT_MODULE = {
    "long_range_navigation": "prompts.uav_navigation",
}


def build_messages(config, prior_results: list = None, feedback: bool = False) -> list[dict]:
    domain = getattr(config, "domain", None)
    prior_results = prior_results or []

    if domain is not None:
        return _build_domain_messages(domain, prior_results, feedback=feedback)

    task_name = config.pipeline.task_name
    mod_path = TASK_TO_PROMPT_MODULE.get(task_name, "prompts.uav_navigation")
    mod = importlib.import_module(mod_path)
    user_content = mod.USER_TEMPLATE
    if feedback:
        fb = _build_feedback_block(prior_results)
        if fb:
            user_content += f"\n\n{fb}"
    return [
        {"role": "system", "content": mod.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_domain_messages(domain, prior_results: list, feedback: bool = False) -> list[dict]:
    mod = importlib.import_module(f"prompts.{domain.name}")

    tensor_ref = getattr(mod, "TENSOR_REFERENCE", None)
    n = len(prior_results)
    variant = ["from_scratch", "refine", "physics_guided"][n % 3]
    few_shot = getattr(mod, "FEW_SHOT_EXAMPLE", "")

    if feedback:
        prior_str = _build_feedback_block(prior_results)
    else:
        prior_str = _format_prior_results(prior_results)

    user_content = mod.USER_TEMPLATE.format(
        task_description=domain.task_description,
        tensor_reference=tensor_ref,
        few_shot=few_shot,
        prior_results=prior_str,
        variant=variant,
    )

    return [
        {"role": "system", "content": mod.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _format_prior_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["\nPREVIOUS ATTEMPTS THIS RUN:"]
    for r in results[-4:]:
        idx = r.get("candidate_index", "?")
        status = r.get("status", "?")
        reason = r.get("reason", "")
        if status == "accepted":
            lines.append(f"  \u2713 Candidate {idx}: ACCEPTED")
        else:
            rt_msg = r.get("runtime_test", {}).get("message", "")
            st_msg = r.get("static_validation", {}).get("message", "")
            lines.append(f"  \u2717 Candidate {idx}: REJECTED \u2014 {reason}")
            if rt_msg:
                lines.append(f"    Runtime error: {rt_msg}")
            if st_msg and st_msg != "OK":
                lines.append(f"    AST error: {st_msg}")
    return "\n".join(lines) + "\n"


def _build_feedback_block(prior_results: list) -> str:
    if not prior_results:
        return ""

    accepted = [r for r in prior_results if r.get("status") == "accepted"]
    rejected = [r for r in prior_results if r.get("status") == "rejected"]
    iteration = len(prior_results)

    lines = ["=" * 60, "FEEDBACK FROM PREVIOUS CANDIDATES", "=" * 60]

    if accepted:
        best = accepted[-1]
        metrics = best.get("runtime_test", {}).get("metrics", {})
        lines += [
            "",
            ">>> BEST ACCEPTED REWARD SO FAR <<<",
            "Build on this \u2014 keep what works, improve what doesn't.",
            "",
            best.get("code", "# (no code available)"),
            "",
        ]
        mean = metrics.get('mean')
        std = metrics.get('std')
        rmin = metrics.get('min')
        rmax = metrics.get('max')
        lines.append(
            f"Runtime metrics: "
            f"mean={f'{mean:.4f}' if mean is not None else 'N/A'}  "
            f"std={f'{std:.4f}' if std is not None else 'N/A'}  "
            f"min={f'{rmin:.4f}' if rmin is not None else 'N/A'}  "
            f"max={f'{rmax:.4f}' if rmax is not None else 'N/A'}"
        )
    else:
        lines += [
            "",
            ">>> NO ACCEPTED REWARD YET <<<",
            "No candidate has passed all checks. Try a different structure.",
        ]

    if rejected:
        recent_rejections = rejected[-3:]
        lines += [
            "",
            ">>> RECENT FAILURES \u2014 DO NOT REPEAT THESE MISTAKES <<<",
        ]
        for r in recent_rejections:
            reason = r.get("reason", "unknown")
            stage = r.get("stage", "unknown")
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

    directive = _build_improvement_directive(accepted, rejected, iteration)
    lines += [
        "",
        ">>> IMPROVEMENT DIRECTIVE FOR THIS ITERATION <<<",
        directive,
    ]

    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


def _build_improvement_directive(accepted: list, rejected: list, iteration: int) -> str:
    directives = []

    if accepted:
        metrics = accepted[-1].get("runtime_test", {}).get("metrics", {})
        mean = metrics.get("mean")
        std = metrics.get("std")

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
                f"Reward std is high ({std:.3f}) \u2014 reward varies wildly across "
                f"environments. Consider clamping components or normalizing scales."
            )

    iteration_goals = {
        0: "Produce a valid reward that passes all shape and validation checks.",
        1: "Improve reward density \u2014 every step should give a non-zero signal.",
        2: "Add a crash/out-of-bounds penalty: penalize Z < 0.1 or Z > 2.0.",
        3: "Add a progress bonus: reward the agent more when actively moving toward goal.",
        4: "Try potential-based shaping: reward = potential(current) - potential(next).",
    }
    goal = iteration_goals.get(iteration, "Improve reward quality and diversity.")
    directives.append(f"This iteration's goal: {goal}")

    if len(accepted) >= 2:
        directives.append(
            "IMPORTANT: Recent accepted rewards share similar structure. "
            "Try a meaningfully different approach \u2014 different components, "
            "different shaping functions, or different scale values."
        )

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
