from pathlib import Path

import pytest

from reward_generator.prompt_loader import load_prompts


def test_load_from_module(tmp_path: Path):
    mod = tmp_path / "prompts.py"
    mod.write_text(
        'TEST_PROMPTS = ["first prompt", "second prompt", "third prompt"]\n',
        encoding="utf-8",
    )
    prompts = load_prompts(mod)
    assert prompts == ["first prompt", "second prompt", "third prompt"]


def test_load_from_module_missing_attribute(tmp_path: Path):
    mod = tmp_path / "prompts.py"
    mod.write_text("MANDATORY = ['self.step_dt']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No prompt list found"):
        load_prompts(mod)


def test_load_from_json(tmp_path: Path):
    path = tmp_path / "prompts.json"
    path.write_text('["one", "two lines\\nhere", "three"]', encoding="utf-8")
    assert load_prompts(path) == ["one", "two lines\nhere", "three"]


def test_load_from_text_one_per_line(tmp_path: Path):
    path = tmp_path / "prompts.txt"
    path.write_text("alpha\n\nbeta\n# not a comment, a prompt\n", encoding="utf-8")
    assert load_prompts(path) == ["alpha", "beta", "# not a comment, a prompt"]


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "nope.py")
