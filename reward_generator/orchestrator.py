from reward_generator.prompt_builder import build_messages
from validators.code_extractor import extract_code_block
from validators.ast_validator import static_validate
from validators.diversity_checker import diversity_check
from validators.runtime_tester import runtime_test
from reward_generator.reward_store import RewardStore
from utils.io import timestamp


class RewardGenerationOrchestrator:
    def __init__(self, llm_client, config):
        self.llm = llm_client
        self.config = config
        self.store = RewardStore("outputs/rewards", "outputs/logs")

    def run(self):
        accepted = []
        all_results = []  # tracks ALL candidates — accepted AND rejected

        for idx in range(self.config.pipeline.num_candidates):

            # --- Build prompt using ALL prior results (accepted + rejected) ---
            messages = build_messages(
                config=self.config,
                prior_results=all_results,
            )

            # --- LLM inference ---
            # Vary temperature per candidate for structural diversity
            temperatures = getattr(self.config.pipeline, "temperatures", None)
            temperature = (
                temperatures[idx % len(temperatures)]
                if temperatures
                else self.config.model.temperature
            )
            raw_output = self.llm.generate(
                messages,
                max_tokens=self.config.model.max_tokens,
                temperature=temperature,
                top_p=self.config.model.top_p,
            )

            record = {
                "candidate_index": idx,
                "temperature_used": temperature,
                "raw_output": raw_output,
                "code": None,
            }

            # ── STAGE 1: Code Extraction ──────────────────────────────────
            code = extract_code_block(raw_output)
            if code is None:
                record["status"] = "rejected"
                record["reason"] = "no_code_block"
                record["stage"] = "code_extractor"
                self.store.save_candidate(
                    f"rejected_{timestamp()}_{idx}", "# no code extracted\n", record
                )
                all_results.append(record)
                continue

            record["code"] = code

            # ── STAGE 2: AST Static Validation ───────────────────────────
            ok_static, msg_static = static_validate(code)
            record["static_validation"] = {"ok": ok_static, "message": msg_static}
            if not ok_static:
                record["status"] = "rejected"
                record["reason"] = "static_validation_failed"
                record["stage"] = "ast_validator"
                self.store.save_candidate(
                    f"rejected_{timestamp()}_{idx}", code, record
                )
                all_results.append(record)
                continue

            # ── STAGE 3: Diversity Check ──────────────────────────────────
            domain = getattr(self.config, "domain", None)
            ref_comps = set(domain.cfg_scales.keys()) if domain else None
            ok_diverse, msg_diverse = diversity_check(code, reference_components=ref_comps)
            record["diversity_check"] = {"ok": ok_diverse, "message": msg_diverse}
            if not ok_diverse:
                record["status"] = "rejected"
                record["reason"] = "too_similar_to_reference"
                record["stage"] = "diversity_checker"
                self.store.save_candidate(
                    f"rejected_{timestamp()}_{idx}", code, record
                )
                all_results.append(record)
                continue

            # ── STAGE 4: Runtime Smoke Test ───────────────────────────────
            ok_runtime, msg_runtime, metrics = runtime_test(
                code,
                batch_size=self.config.runtime_test.batch_size,
            )
            record["runtime_test"] = {
                "ok": ok_runtime,
                "message": msg_runtime,
                "metrics": metrics,
            }
            if not ok_runtime:
                record["status"] = "rejected"
                record["reason"] = "runtime_test_failed"
                record["stage"] = "runtime_tester"
                self.store.save_candidate(
                    f"rejected_{timestamp()}_{idx}", code, record
                )
                all_results.append(record)
                continue

            # ── ACCEPTED ─────────────────────────────────────────────────
            record["status"] = "accepted"
            record["reason"] = "passed_all_checks"
            record["stage"] = "done"
            name = f"reward_{timestamp()}_{idx}"
            code_path, meta_path = self.store.save_candidate(name, code, record)
            accepted.append({
                "code_path": code_path,
                "meta_path": meta_path,
                "metrics": metrics,
            })
            all_results.append(record)

        self._print_summary(all_results)
        return accepted

    def _print_summary(self, all_results: list):
        total = len(all_results)
        accepted = sum(1 for r in all_results if r["status"] == "accepted")
        print(f"\n{'='*50}")
        print(f"  Generation Summary")
        print(f"{'='*50}")
        print(f"  Total candidates : {total}")
        print(f"  Accepted         : {accepted}")
        print(f"  Rejected         : {total - accepted}")
        print(f"{'─'*50}")
        stage_counts = {}
        for r in all_results:
            if r["status"] == "rejected":
                stage = r.get("reason", "unknown")
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        for reason, count in stage_counts.items():
            print(f"  ✗ {reason:<35} {count}")
        print(f"{'='*50}\n")