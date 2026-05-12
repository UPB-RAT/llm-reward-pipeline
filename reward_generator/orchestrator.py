from reward_generator.prompt_builder import build_messages
from validators.code_extractor import extract_code
from validators.ast_validator import validate_ast
from validators.diversity_checker import diversity_check
from validators.runtime_tester import runtime_test
from reward_generator.reward_store import RewardStore
from utils.io import timestamp
import re


def _extract_forbidden_attrs(error_msg: str) -> list[str]:
    return re.findall(r"has no attribute '([^']+)'", error_msg)

def _extract_shape_error(error_msg: str) -> str | None:
    if "size" in error_msg or "shape" in error_msg or "dimension" in error_msg:
        return error_msg
    return None

def _build_failure_patch(all_results: list[dict]) -> str:
    forbidden_attrs: set[str] = set()
    shape_errors: set[str] = set()
    for r in all_results:
        if r.get("status") != "rejected":
            continue
        rt = r.get("runtime_test", {})
        if not rt.get("ok", True):
            msg = rt.get("message", "")
            forbidden_attrs.update(_extract_forbidden_attrs(msg))
            shape_err = _extract_shape_error(msg)
            if shape_err:
                shape_errors.add(shape_err)
    if not forbidden_attrs and not shape_errors:
        return ""
    lines = ["\n⚠️  PREVIOUS ATTEMPTS FAILED — avoid these exact mistakes:\n"]
    if forbidden_attrs:
        lines.append("FORBIDDEN (these attributes do NOT exist in the env):")
        for attr in sorted(forbidden_attrs):
            lines.append(f"  ✗  self.{attr}")
        lines.append("")
    if shape_errors:
        lines.append("SHAPE ERRORS seen in previous attempts:")
        for err in sorted(shape_errors):
            lines.append(f"  ✗  {err}")
        lines.append("")
    lines.append("Only use the ALLOWED ATTRIBUTES listed above. Nothing else.")
    return "\n".join(lines)


class RewardGenerationOrchestrator:

    def __init__(self, llm_client, config):
        self.llm    = llm_client
        self.config = config
        self.store  = RewardStore("outputs/rewards", "outputs/logs")

    def run(self):
        accepted    = []
        all_results = []

        for idx in range(self.config.pipeline.num_candidates):

            failure_patch = _build_failure_patch(all_results)

            messages = build_messages(
                domain        = self.config.domain,
                prior_results = all_results,
                failure_patch = failure_patch,
            )

            temperatures = getattr(self.config.pipeline, "temperatures", None)
            temperature  = (
                temperatures[idx % len(temperatures)]
                if temperatures else self.config.model.temperature
            )

            raw_output = self.llm.generate(
                messages,
                max_tokens  = self.config.model.max_tokens,
                temperature = temperature,
                top_p       = self.config.model.top_p,
            )

            record = dict(
                candidate_index  = idx,
                temperature_used = temperature,
                raw_output       = raw_output,
                code             = None,
            )

            # Stage 1: Code extraction
            code = extract_code(raw_output)
            if not isinstance(code, str) or not code.strip():
                record.update(status="rejected", reason="no_code_block",
                               stage="code_extractor")
                self.store.save_candidate(f"rejected_{timestamp()}_{idx}", "", record)
                all_results.append(record)
                continue
            record["code"] = code

            # Stage 2: AST static validation
            ok_static, msg_static = validate_ast(code)
            record["static_validation"] = {"ok": ok_static, "message": msg_static}
            if not ok_static:
                record.update(status="rejected", reason="static_validation_failed",
                               stage="ast_validator")
                self.store.save_candidate(f"rejected_{timestamp()}_{idx}", code, record)
                all_results.append(record)
                continue

            # Stage 3: Diversity check
            ok_diverse, msg_diverse = diversity_check(code, self.config.domain)
            record["diversity_check"] = {"ok": ok_diverse, "message": msg_diverse}
            if not ok_diverse:
                record.update(status="rejected", reason="too_similar_to_reference",
                               stage="diversity_checker")
                self.store.save_candidate(f"rejected_{timestamp()}_{idx}", code, record)
                all_results.append(record)
                continue

            # Stage 4: Runtime smoke test
            ok_runtime, msg_runtime, metrics = runtime_test(
                code,
                batch_size = self.config.runtime_test.batch_size,
                domain     = self.config.domain,
            )
            record["runtime_test"] = {"ok": ok_runtime, "message": msg_runtime,
                                       "metrics": metrics}
            if not ok_runtime:
                record.update(status="rejected", reason="runtime_test_failed",
                               stage="runtime_tester")
                self.store.save_candidate(f"rejected_{timestamp()}_{idx}", code, record)
                all_results.append(record)
                continue

            # Accepted
            record.update(status="accepted", reason="passed_all_checks", stage="done")
            name = f"reward_{timestamp()}_{idx}"
            code_path, meta_path = self.store.save_candidate(name, code, record)
            accepted.append(dict(code_path=code_path, meta_path=meta_path,
                                  metrics=metrics))
            all_results.append(record)

        self._print_summary(all_results)
        self._print_rewards(all_results)   # ← print all reward functions
        return accepted

    # ── Summary table ─────────────────────────────────────────────────────────
    def _print_summary(self, all_results: list):
        total    = len(all_results)
        accepted = sum(1 for r in all_results if r["status"] == "accepted")
        print("=" * 50)
        print("  Generation Summary")
        print("=" * 50)
        print(f"  Total candidates : {total}")
        print(f"  Accepted         : {accepted}")
        print(f"  Rejected         : {total - accepted}")
        print("-" * 50)
        stage_counts: dict[str, int] = {}
        for r in all_results:
            if r["status"] == "rejected":
                stage = r.get("reason", "unknown")
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        for reason, count in stage_counts.items():
            print(f"  ✗ {reason:<35} {count}")
        print("=" * 50)

    # ── Print all reward functions ─────────────────────────────────────────────
    def _print_rewards(self, all_results: list):
        accepted = [r for r in all_results if r.get("status") == "accepted"]
        rejected = [r for r in all_results if r.get("status") == "rejected"]

        if accepted:
            print("\n" + "=" * 60)
            print(f"  ✅  ACCEPTED REWARD FUNCTIONS ({len(accepted)})")
            print("=" * 60)
            for i, r in enumerate(accepted, 1):
                print(f"\n── Accepted #{i}  "
                      f"(candidate {r.get('candidate_index')}, "
                      f"temp={r.get('temperature_used')})")
                print("-" * 60)
                print(r.get("code", "<no code>"))

        if rejected:
            print("\n" + "=" * 60)
            print(f"  ❌  REJECTED REWARD FUNCTIONS ({len(rejected)})")
            print("=" * 60)
            for i, r in enumerate(rejected, 1):
                code   = r.get("code") or "<no code extracted>"
                reason = r.get("reason", "unknown")
                stage  = r.get("stage",  "unknown")
                rt_msg = r.get("runtime_test", {}).get("message", "")
                st_msg = r.get("static_validation", {}).get("message", "")
                print(f"\n── Rejected #{i}  "
                      f"(candidate {r.get('candidate_index')}, "
                      f"temp={r.get('temperature_used')})")
                print(f"   Reason : {reason}  [{stage}]")
                if rt_msg:
                    print(f"   Error  : {rt_msg}")
                if st_msg and st_msg != "OK":
                    print(f"   AST    : {st_msg}")
                print("-" * 60)
                print(code)

        print("\n" + "=" * 60)