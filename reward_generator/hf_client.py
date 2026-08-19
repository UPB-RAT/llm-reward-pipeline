import sys
from pathlib import Path


class HFLLMClient:
    def __init__(self, base_model_path: str, adapter_path: str | None = None, **kwargs):
        self.base_model_path = str(base_model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None

        try:
            from vllm import LLM
        except ImportError as e:
            print(
                f"❌ ERROR: Missing required library. Please install vllm: {e}",
                file=sys.stderr,
            )
            raise e

        for path_val, name in [
            (self.base_model_path, "Base model"),
            (self.adapter_path, "Adapter"),
        ]:
            if path_val and ("/" in path_val and not path_val.startswith(("./", "/"))):
                continue
            if path_val and not Path(path_val).exists():
                raise FileNotFoundError(f"{name} path not found: {path_val}")

        llm_kwargs = dict(
            model=self.base_model_path,
            trust_remote_code=True,
        )

        if self.adapter_path:
            llm_kwargs["enable_lora"] = True

        print(f"🤖 Loading vLLM Model: {self.base_model_path}...")
        self.llm = LLM(**llm_kwargs)
        self._lora_request = None
        if self.adapter_path:
            from vllm.lora.request import LoRARequest

            print(f"🔌 Loading LoRA Adapter: {self.adapter_path}...")
            self._lora_request = LoRARequest("adapter", 1, self.adapter_path)

    def generate(self, messages, max_tokens=2048, temperature=0.7, top_p=0.95):
        from vllm import SamplingParams

        try:
            do_sample = temperature > 0.0
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
                top_k=40 if do_sample else -1,
                repetition_penalty=1.1,
            )

            outputs = self.llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                lora_request=self._lora_request,
            )

            content = outputs[0].outputs[0].text
            finish_reason = outputs[0].outputs[0].finish_reason
            if finish_reason == "length":
                print(
                    f"  ⚠️  WARNING: generation stopped at max_tokens limit ({max_tokens})."
                )
            return content
        except Exception as e:
            print(f"❌ Generation error: {e}", file=sys.stderr)
            raise e
