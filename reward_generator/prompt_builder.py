import importlib
from pathlib import Path


TASK_TO_PROMPT_MODULE = {
    "long_range_navigation": "prompts.uav_navigation",
}


def build_messages(config, prior_results: list = None) -> list[dict]:
    domain = getattr(config, "domain", None)
    prior_results = prior_results or []

    if domain is not None:
        return _build_domain_messages(domain, prior_results)

    task_name = config.pipeline.task_name
    mod_path = TASK_TO_PROMPT_MODULE.get(task_name, "prompts.uav_navigation")
    mod = importlib.import_module(mod_path)
    return [
        {"role": "system", "content": mod.SYSTEM_PROMPT},
        {"role": "user", "content": mod.USER_TEMPLATE},
    ]


def _build_domain_messages(domain, prior_results: list) -> list[dict]:
    mod = importlib.import_module(f"prompts.{domain.name}")

    tensor_ref = getattr(mod, "TENSOR_REFERENCE", None) or _build_tensor_reference(domain)
    baseline_code = Path(domain.env_reference_path).read_text()
    prior_str = _format_prior_results(prior_results)
    n = len(prior_results)
    variant = ["from_scratch", "refine", "physics_guided"][n % 3]
    few_shot = getattr(mod, "FEW_SHOT_EXAMPLE", "")

    system_parts = [mod.SYSTEM_PROMPT]
    constraints = _build_constraints(domain)
    system_parts.append(constraints)
    full_system = "\n".join(system_parts)

    user_content = mod.USER_TEMPLATE.format(
        task_description=domain.task_description,
        tensor_reference=tensor_ref,
        few_shot=few_shot,
        baseline_code=baseline_code,
        prior_results=prior_str,
        variant=variant,
    )

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_content},
    ]


def _build_tensor_reference(domain) -> str:
    tensor_shapes = getattr(domain, "tensor_shapes", {})
    cfg_scales = getattr(domain, "cfg_scales", {})

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
    lines.append(f"SCALAR: self.step_dt = {domain.step_dt}")
    lines.append("")
    lines.append("NOTHING ELSE EXISTS. No self.num_envs, obs_dict, cfg[\"...\"], device.")

    return "\n".join(lines)


def _build_constraints(domain) -> str:
    tensor_shapes = getattr(domain, "tensor_shapes", {})
    cfg_scales = getattr(domain, "cfg_scales", {})

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
        f"  step_dt    = {domain.step_dt}",
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
