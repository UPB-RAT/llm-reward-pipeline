import ast

# Component names present in the baseline IsaacLab reference reward
REFERENCE_COMPONENTS = {
    "lin_vel",
    "ang_vel",
    "distance_to_goal",
    "distance_to_goal_mapped",
}

# Minimum number of NEW variable names beyond the reference
NOVELTY_REQUIRED = 2


def diversity_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors are handled by the AST validator — skip here
        return True, "skip — syntax error handled by ast_validator"

    # Collect all assigned variable names in the function body
    assigned_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, (ast.AnnAssign,)):
            if isinstance(node.target, ast.Name):
                assigned_names.add(node.target.id)

    # Filter out noise: single-letter names, 'reward', 'rewards', 'self'
    noise = {"reward", "rewards", "self", "key", "value", "i", "n"}
    assigned_names -= noise

    # Count names that are NOT in the reference
    novel_names = assigned_names - REFERENCE_COMPONENTS
    novel_count = len(novel_names)

    if novel_count < NOVELTY_REQUIRED:
        return (
            False,
            f"Too similar to reference — only {novel_count} novel variable(s) found: "
            f"{novel_names}. Need at least {NOVELTY_REQUIRED}.",
        )

    return True, f"Diverse — {novel_count} novel component(s): {novel_names}"