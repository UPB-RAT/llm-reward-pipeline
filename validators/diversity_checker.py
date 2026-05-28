import ast
import pathlib


MIN_NOVEL = 2   # minimum new reward component variables beyond reference baseline

# Names to always exclude from novelty count
_BUILTINS = {
    # Python built-ins
    "list", "dict", "set", "tuple", "int", "float", "bool", "str",
    "len", "range", "zip", "map", "filter", "enumerate", "print",
    "True", "False", "None", "type", "isinstance", "hasattr", "getattr",
    # torch namespace
    "torch",
    # mandatory structural names every _get_rewards must have
    "rewards", "total_reward",
    # common loop/temp vars
    "k", "v", "i", "n", "x", "y", "z",
}


def _get_reward_variable_names(code: str) -> set[str]:
    """
    Extract only assignment target names (i.e. variables the LLM *defines*),
    not all Name references. This avoids counting built-ins, dict keys, and
    function calls as novel components.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names = set()
    for node in ast.walk(tree):
        # regular assignments: x = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        # annotated assignments: x: float = ...
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)

    return names


def _load_reference_names(env_reference_path: str) -> set[str]:
    ref_code = pathlib.Path(env_reference_path).read_text()
    return _get_reward_variable_names(ref_code)


def diversity_check(code: str, domain) -> tuple[bool, str]:
    reference_names = _load_reference_names(domain.env_reference_path)
    candidate_names = _get_reward_variable_names(code)
    novel = candidate_names - reference_names - _BUILTINS
    # keep only reward component names: must be longer than 3 chars,
    # no leading underscore, and look like a reward term (start with rew_)
    novel_rew = {n for n in novel if n.startswith("rew_")}
    # fallback: also accept any long non-private novel name
    novel_other = {n for n in novel if len(n) > 4 and not n.startswith("_") and not n.startswith("rew_")}
    novel_all = novel_rew | novel_other

    if len(novel_all) < MIN_NOVEL:
        return False, f"Too similar: only {len(novel_all)} novel components ({novel_all})"
    return True, f"Diverse: {len(novel_all)} novel components {novel_all}"