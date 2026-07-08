import re


def extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if "def _get_rewards" in stripped:
        return stripped
    return None