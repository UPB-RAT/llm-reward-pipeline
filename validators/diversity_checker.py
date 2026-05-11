import ast
import pathlib

MIN_NOVEL = 2   # minimum new variables beyond reference baseline


def _get_variable_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _load_reference_names(env_reference_path: str) -> set[str]:
    ref_code = pathlib.Path(env_reference_path).read_text()
    return _get_variable_names(ref_code)


def diversity_check(code: str, domain) -> tuple[bool, str]:
    reference_names = _load_reference_names(domain.env_reference_path)
    candidate_names = _get_variable_names(code)
    novel = candidate_names - reference_names
    # filter out short/generic names
    novel = {n for n in novel if len(n) > 3 and not n.startswith("_")}
    if len(novel) < MIN_NOVEL:
        return False, f"Too similar: only {len(novel)} novel components ({novel})"
    return True, f"Diverse: {len(novel)} novel components {novel}"