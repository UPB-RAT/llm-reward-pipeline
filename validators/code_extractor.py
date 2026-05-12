"""
validators/code_extractor.py  — drop-in replacement
Robustly extracts the first ```python ... ``` block from raw LLM output.
Falls back to extracting bare def _get_rewards(...) if no fenced block found.
"""
import re


def extract_code(raw_output: str) -> str | None:
    """
    Extract Python code from LLM output.

    Priority:
    1. First ```python ... ``` fenced block
    2. First ``` ... ``` fenced block (language tag missing)
    3. Bare function starting with 'def _get_rewards'
    Returns None if nothing found.
    """
    if not isinstance(raw_output, str):
        return None

    # 1. ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. ``` ... ``` (no language tag)
    match = re.search(r"```\s*\n(.*?)```", raw_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            return code

    # 3. Bare function — everything from 'def _get_rewards' to end
    match = re.search(r"(def _get_rewards\(.*)", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None