from reward_generator.config import (
    AppConfig,
    DomainConfig,
    ModelConfig,
    PipelineConfig,
    RuntimeTestConfig,
)
from reward_generator.prompt_builder import (
    _build_failure_patch,
    _clean_error,
    build_messages,
)


def _make_config(task_name="quadcopter", prompt_style="detailed", feedback=False, with_domain=True):
    pipeline = PipelineConfig(
        num_candidates=5,
        task_name=task_name,
        save_all_raw=False,
        random_seed=42,
        feedback=feedback,
        prompt_style=prompt_style,
    )
    model = ModelConfig(temperature=0.7, top_p=0.95, max_tokens=2048, n_ctx=8192, n_gpu_layers=-1)
    runtime = RuntimeTestConfig(batch_size=16, max_episode_len=500)
    domain = None
    if with_domain:
        domain = DomainConfig(
            name="quadcopter",
            task_description="Quadcopter hover and navigation task.",
            env_reference_path="envs/quadcopter_env_reference.py",
            step_dt=0.02,
            cfg_scales={"lin_vel_reward_scale": -0.05},
            tensor_shapes={"_robot.data.root_pos_w": [16, 3]},
        )
    return AppConfig(model=model, pipeline=pipeline, runtime_test=runtime, domain=domain)


def test_experimental_style_returns_user_only_message():
    config = _make_config(prompt_style="experimental")
    messages = build_messages(config, prior_results=[])

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Design a reward function for a quadcopter hover-and-navigation task" in messages[0]["content"]
    assert "self._robot.data.root_pos_w" in messages[0]["content"]
    assert "self.step_dt" in messages[0]["content"]
    assert "from_scratch" in messages[0]["content"]
    assert "Return ONLY the raw Python code" in messages[0]["content"]


def test_experimental_style_with_feedback():
    config = _make_config(prompt_style="experimental", feedback=True)
    prior = [
        {
            "candidate_index": 0,
            "status": "rejected",
            "reason": "runtime_test_failed",
            "stage": "runtime_tester",
            "code": "def _get_rewards(self):\n    pass",
            "runtime_test": {"ok": False, "message": "Shape mismatch — expected (16,), got (16, 3)"},
        }
    ]
    messages = build_messages(config, prior_results=prior, feedback=True)

    assert messages[0]["role"] == "user"
    assert "PREVIOUS ATTEMPTS FAILED" in messages[0]["content"]
    assert "every component must be [N]" in messages[0]["content"]


def test_detailed_domain_injects_failure_patch_into_system_prompt():
    config = _make_config(prompt_style="detailed", feedback=True)
    prior = [
        {
            "candidate_index": 0,
            "status": "rejected",
            "reason": "runtime_test_failed",
            "stage": "runtime_tester",
            "code": "def _get_rewards(self):\n    return torch.zeros(16)",
            "runtime_test": {"ok": False, "message": "'_DummyEnv' object has no attribute 'ground_truth'" },
        }
    ]
    messages = build_messages(config, prior_results=prior, feedback=True)

    assert messages[0]["role"] == "system"
    assert "PREVIOUS ATTEMPTS FAILED" in messages[0]["content"]
    assert "does NOT exist in the sandbox" in messages[0]["content"]
    assert "self._robot.data.root_pos_w" in messages[0]["content"]
    assert "ground_truth" not in messages[0]["content"]
    assert "_DummyEnv" not in messages[0]["content"]


def test_detailed_domain_without_feedback_no_failure_patch():
    config = _make_config(prompt_style="detailed", feedback=False)
    prior = [
        {
            "candidate_index": 0,
            "status": "rejected",
            "reason": "runtime_test_failed",
            "runtime_test": {"ok": False, "message": "Dimension out of range"},
        }
    ]
    messages = build_messages(config, prior_results=prior, feedback=False)

    assert "PREVIOUS ATTEMPTS FAILED" not in messages[0]["content"]
    assert "PREVIOUS ATTEMPTS THIS RUN:" in messages[1]["content"]


def test_detailed_style_domain_unchanged():
    config = _make_config(prompt_style="detailed")
    messages = build_messages(config, prior_results=[])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Quadcopter hover and navigation task." in messages[1]["content"]
    assert "domain_constraints" in messages[1]["content"]


def test_detailed_style_task_path():
    config = _make_config(task_name="long_range_navigation", with_domain=False)
    messages = build_messages(config, prior_results=[])

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert "long-range navigation" in messages[1]["content"]


def test_unknown_style_raises():
    config = _make_config(prompt_style="nonexistent")
    try:
        build_messages(config, prior_results=[])
    except ValueError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown prompt_style")


def test_prompt_override_uses_exact_text():
    config = _make_config(prompt_style="detailed", feedback=True)
    messages = build_messages(
        config,
        prior_results=[{"status": "rejected"}],
        feedback=True,
        prompt_override="Custom test prompt text.",
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Custom test prompt text."


def test_failure_patch_skips_early_stage_rejections():
    results = [
        {"status": "rejected", "reason": "no_code_block", "stage": "code_extractor"},
        {
            "status": "rejected",
            "reason": "static_validation_failed",
            "static_validation": {"ok": False, "message": "Forbidden import"},
        },
        {
            "status": "rejected",
            "reason": "runtime_test_failed",
            "runtime_test": {"ok": False, "message": "Wrong output shape"},
        },
    ]
    patch = _build_failure_patch(results)
    assert "PREVIOUS ATTEMPTS FAILED" in patch
    assert "returned a reward tensor of the wrong shape" in patch
    assert "FORBIDDEN" not in patch


def test_failure_patch_counts_categories_without_repeating_attribute_names():
    results = []
    for i in range(5):
        results.append({
            "candidate_index": i,
            "status": "rejected",
            "reason": "runtime_test_failed",
            "runtime_test": {"ok": False, "message": f"'Env' object has no attribute 'dream{i}'"},
        })
    patch = _build_failure_patch(results)
    assert "PREVIOUS ATTEMPTS FAILED" in patch
    assert "seen 5\u00d7" in patch
    assert "does NOT exist in the sandbox" in patch
    assert "dream" not in patch


def test_prior_results_sanitizes_runtime_error_tokens():
    config = _make_config(prompt_style="detailed", feedback=False)
    prior = [
        {
            "candidate_index": 3,
            "status": "rejected",
            "reason": "runtime_test_failed",
            "runtime_test": {"ok": False, "message": "'_Data' object has no attribute 'joint_torques'"},
        }
    ]
    messages = build_messages(config, prior_results=prior, feedback=False)

    user_content = messages[1]["content"]
    assert "PREVIOUS ATTEMPTS THIS RUN:" in user_content
    assert "accessed a `self.*` attribute that does not exist" in user_content
    assert "joint_torques" not in user_content
    assert "_Data" not in user_content


def test_clean_error_strips_signature_dump():
    msg = "Runtime call failed: torch.broadcast_tensors(expected one of:\n  (Tensor, Tensor)\n  ...)"
    cleaned = _clean_error(msg)
    assert "expected one of:" not in cleaned
    assert cleaned == "Runtime call failed: torch.broadcast_tensors"
