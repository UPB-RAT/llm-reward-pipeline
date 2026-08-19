import importlib
import re

try:
    from string import Formatter
except ImportError:  # pragma: no cover
    Formatter = None


TASK_TO_PROMPT_MODULE = {
    "long_range_navigation": "prompts.uav_navigation",
}

DEFAULT_PROMPT_STYLE = "detailed"

PROMPT_STYLE_MODULES = {
    "experimental": "prompts.experimental",
}

# Token-budget guards for feedback (Channel A failure scorecard + Channel B window).
PRIOR_RESULTS_WINDOW = 4
CODE_PREVIEW_LINES = 6

# Runtime failure categories distilled into token-free feedback.
# Order matters: the first matching category wins for a given error message.
_RUNTIME_CATEGORY_PATTERNS = [
    ("missing_attr", re.compile(r"has no attribute", re.IGNORECASE)),
    ("dim_out_of_range", re.compile(r"dimension out of range", re.IGNORECASE)),
    ("shape_mismatch", re.compile(r"shape mismatch|bad components", re.IGNORECASE)),
    ("invalid_output", re.compile(r"nan in reward|inf in reward|not a torch\.tensor", re.IGNORECASE)),
    ("output_shape", re.compile(r"wrong output shape", re.IGNORECASE)),
]

# Channel A — sentence per category. Deliberately token-free: we never repeat
# hallucinated attribute names or raw torch messages back into the prompt,
# because naming a plausible-but-wrong token anchors the model and makes it
# repeat the mistake.
_RUNTIME_CATEGORY_TEXT = {
    "missing_attr": "attempts referenced a `self.*` attribute that does NOT exist in the sandbox — use ONLY the listed attributes",
    "dim_out_of_range": "attempts crashed with \"Dimension out of range\" — do NOT pass `dim=` to elementwise ops on 1-D slices; reduce [N, 3] tensors with `dim=1` only",
    "shape_mismatch": "attempts produced reward components with the wrong shape — every component must be [N]; avoid `.mean()` and boolean-indexing that collapses shape",
    "invalid_output": "attempts returned NaN or Inf rewards — keep outputs as [N] float32 and avoid degenerate divisions",
    "output_shape": "attempts returned a reward tensor of the wrong shape — it must be exactly [N]",
    "other_runtime": "attempts hit other runtime errors — follow the required structure exactly",
}

# Channel B — short sanitized label for a rejected candidate's runtime failure.
_CATEGORY_SHORT_LABEL = {
    "missing_attr": "accessed a `self.*` attribute that does not exist",
    "dim_out_of_range": "\"Dimension out of range\" (dim passed to a 1-D slice)",
    "shape_mismatch": "a reward component had the wrong shape",
    "invalid_output": "returned NaN or Inf",
    "output_shape": "returned a reward of the wrong shape",
    "other_runtime": "other runtime error",
}

_TOKEN_STRIP_RE = re.compile(r"'\s*[^']*?'")


def build_messages(
    config,
    prior_results: list = None,
    feedback: bool = False,
    prompt_override: str | None = None,
) -> list[dict]:
    if prompt_override is not None:
        content = prompt_override
        if feedback:
            prior_results = prior_results or []
            patch = _build_failure_patch(prior_results)
            if patch:
                content += f"\n\n{patch}"
        return [{"role": "user", "content": content}]

    prior_results = prior_results or []
    style = getattr(config.pipeline, "prompt_style", DEFAULT_PROMPT_STYLE)

    if style == "experimental":
        return _build_experimental_messages(prior_results, feedback=feedback)
    if style != DEFAULT_PROMPT_STYLE:
        raise ValueError(
            f"Unknown prompt_style: '{style}'. "
            f"Supported styles: {[DEFAULT_PROMPT_STYLE, *PROMPT_STYLE_MODULES]}."
        )

    domain = getattr(config, "domain", None)
    if domain is not None:
        return _build_domain_messages(domain, prior_results, feedback=feedback)

    task_name = config.pipeline.task_name
    mod_path = TASK_TO_PROMPT_MODULE.get(task_name, "prompts.uav_navigation")
    mod = importlib.import_module(mod_path)
    user_content = mod.USER_TEMPLATE
    if feedback:
        patch = _build_failure_patch(prior_results)
        if patch:
            user_content += f"\n\n{patch}"
    return [
        {"role": "system", "content": mod.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_experimental_messages(prior_results: list, feedback: bool = False) -> list[dict]:
    mod = importlib.import_module(PROMPT_STYLE_MODULES["experimental"])

    user_content = _format_template(
        getattr(mod, "USER_TEMPLATE", ""),
        variant=["from_scratch", "refine", "physics_guided"][len(prior_results) % 3],
    )

    if feedback:
        patch = _build_failure_patch(prior_results)
        if patch:
            user_content += f"\n\n{patch}"

    system_prompt = getattr(mod, "SYSTEM_PROMPT", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _format_template(template: str, **kwargs) -> str:
    """Format a template if it contains placeholders; otherwise return it verbatim."""
    if Formatter is not None:
        field_names = [fn for _, fn, _, _ in Formatter().parse(template) if fn is not None]
        if not field_names:
            return template
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError, IndexError):
        return template


def _build_domain_messages(domain, prior_results: list, feedback: bool = False) -> list[dict]:
    mod = importlib.import_module(f"prompts.{domain.name}")

    tensor_ref = getattr(mod, "TENSOR_REFERENCE", None)
    n = len(prior_results)
    variant = ["from_scratch", "refine", "physics_guided"][n % 3]
    few_shot = getattr(mod, "FEW_SHOT_EXAMPLE", "")

    # Channel B — prior results window into the user prompt (always rendered).
    prior_str = _format_prior_results(prior_results)

    user_content = mod.USER_TEMPLATE.format(
        task_description=domain.task_description,
        tensor_reference=tensor_ref,
        few_shot=few_shot,
        prior_results=prior_str,
        variant=variant,
    )

    # Channel A — token-free failure scorecard into the system prompt (--feedback only).
    system_prompt = mod.SYSTEM_PROMPT
    if feedback:
        patch = _build_failure_patch(prior_results)
        if patch:
            system_prompt = system_prompt.rstrip() + "\n\n" + patch

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _format_prior_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["\nPREVIOUS ATTEMPTS THIS RUN:"]
    for r in results[-PRIOR_RESULTS_WINDOW:]:
        idx = r.get("candidate_index", "?")
        status = r.get("status", "?")
        if status == "accepted":
            lines.append(f"  \u2713 Candidate {idx}: ACCEPTED")
            code = r.get("code")
            if code:
                preview = "\n".join(code.splitlines()[:CODE_PREVIEW_LINES])
                lines.append(f"  ```python\n{preview}\n  ```")
        else:
            reason = r.get("reason", "")
            lines.append(f"  \u2717 Candidate {idx}: REJECTED \u2014 {reason}")
            rt = r.get("runtime_test") or {}
            st = r.get("static_validation") or {}
            rt_msg = rt.get("message", "") if isinstance(rt, dict) else ""
            st_msg = st.get("message", "") if isinstance(st, dict) else ""
            if rt_msg:
                cat = _categorize_runtime_failure(rt_msg)
                if cat == "other_runtime":
                    hint = _strip_tokens(_clean_error(rt_msg))
                    lines.append(f"    Runtime error: {hint or _CATEGORY_SHORT_LABEL[cat]}")
                else:
                    lines.append(f"    {_CATEGORY_SHORT_LABEL[cat]}")
            if st_msg and st_msg != "OK":
                lines.append(f"    AST error: {_strip_tokens(_clean_error(st_msg))}")
    return "\n".join(lines) + "\n"


def _build_failure_patch(results: list[dict]) -> str:
    """Channel A — token-free error scorecard distilled from runtime failures.

    Only records that reached the runtime smoke test and failed contribute.
    Failure *categories* are counted and described WITHOUT repeating the
    hallucinated attribute names or raw torch messages back into the prompt
    (repeating bad tokens anchors the model into repeating them).
    """
    counts: dict[str, int] = {}
    for r in results:
        rt = r.get("runtime_test", {})
        if not isinstance(rt, dict) or rt.get("ok", True):
            continue
        cat = _categorize_runtime_failure(rt.get("message", ""))
        counts[cat] = counts.get(cat, 0) + 1

    if not counts:
        return ""

    lines = ["\u26a0\ufe0f PREVIOUS ATTEMPTS FAILED \u2014 avoid these mistake classes:"]
    for cat, _ in _RUNTIME_CATEGORY_PATTERNS:
        c = counts.get(cat)
        if c:
            lines.append(f"  \u2717 {_RUNTIME_CATEGORY_TEXT[cat]} (seen {c}\u00d7)")
    c = counts.get("other_runtime")
    if c:
        lines.append(f"  \u2717 {_RUNTIME_CATEGORY_TEXT['other_runtime']} (seen {c}\u00d7)")

    allowed = _allowed_attribute_surface()
    if allowed:
        lines.append(
            "The sandbox exposes EXACTLY these attributes: "
            + ", ".join(f"`{a}`" for a in allowed)
            + ". Use ONLY these."
        )
    return "\n".join(lines)


def _categorize_runtime_failure(msg: str) -> str:
    if not isinstance(msg, str) or not msg:
        return "other_runtime"
    for cat, pat in _RUNTIME_CATEGORY_PATTERNS:
        if pat.search(msg):
            return cat
    return "other_runtime"


def _strip_tokens(msg: str) -> str:
    """Remove single-quoted tokens (hallucinated attrs, class names) from a message."""
    if not isinstance(msg, str):
        return ""
    stripped = _TOKEN_STRIP_RE.sub("", msg)
    return re.sub(r"\s{2,}", " ", stripped).strip()


def _allowed_attribute_surface() -> list[str]:
    try:
        from validators.runtime_tester import allowed_attributes as _fn
        return _fn()
    except Exception:  # pragma: no cover - sandbox surface always importable in practice
        return []


def _clean_error(msg: str) -> str:
    """Truncate a torch error: drop signature dumps, keep first 2 lines."""
    if not isinstance(msg, str):
        return ""
    cleaned = re.split(r"expected one of:", msg, flags=re.DOTALL)[0].rstrip(" (")
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()][:2]
    return " | ".join(lines)
