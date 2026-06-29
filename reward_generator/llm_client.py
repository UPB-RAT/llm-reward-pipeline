from pathlib import Path
import sys


class LocalLLMClient:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ):
        base_model_path = str(base_model_path)
        adapter_path = str(adapter_path) if adapter_path else None

        if not Path(base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")

        if adapter_path is not None and not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            print(f"❌ ERROR: Missing required library 'llama-cpp-python'. Please install it: {e}", file=sys.stderr)
            raise e

        llama_kwargs = dict(
            model_path=base_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format="chatml",
            verbose=False,
        )

        if adapter_path is not None:
            llama_kwargs["lora_path"] = adapter_path

        self.llm = Llama(**llama_kwargs)

    def generate(self, messages, max_tokens=2048, temperature=0.7, top_p=0.95):
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["\n\n\n", "\n\n"],
        )
        content = resp["choices"][0]["message"]["content"]
        finish_reason = resp["choices"][0]["finish_reason"]

        if finish_reason == "length":
            print(
                f"  ⚠️  WARNING: generation stopped at max_tokens limit "
                f"({max_tokens}) — output may be incomplete. "
                f"Increase max_tokens or shorten the prompt."
            )

        return content