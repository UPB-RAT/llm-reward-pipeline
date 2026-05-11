# ============================================================
# code_extractor.py  —  DOMAIN-AGNOSTIC CODE EXTRACTOR
# ============================================================
import re

_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def extract_code(raw: str) -> tuple[str | None, bool, str]:
    match = _BLOCK_RE.search(raw)
    if not match:
        return None, False, "No Python code block found in LLM output"
    return match.group(1).strip(), True, "OK"