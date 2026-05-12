from llama_cpp import Llama


class LocalLLMClient:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        self.llm = Llama(
            model_path   = model_path,
            n_ctx        = n_ctx,
            n_gpu_layers = n_gpu_layers,
            chat_format  = "chatml",
            verbose      = False,
        )

    def generate(self, messages, max_tokens=2048, temperature=0.7, top_p=0.95):
        resp = self.llm.create_chat_completion(
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
            stop        = ["```\n\n", "```\n"],   # stop cleanly after closing code fence
        )
        content       = resp["choices"][0]["message"]["content"]
        finish_reason = resp["choices"][0]["finish_reason"]

        # Warn if model hit token limit before finishing
        if finish_reason == "length":
            print(f"  ⚠️  WARNING: generation stopped at max_tokens limit "
                  f"({max_tokens}) — output may be incomplete. "
                  f"Increase max_tokens or shorten the prompt.")

        return content