import sys
from pathlib import Path

class HFLLMClient:
    def __init__(self, base_model_path: str, adapter_path: str | None = None, **kwargs):
        self.base_model_path = str(base_model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError as e:
            print(f"❌ ERROR: Missing required library. Please install transformers/peft/torch: {e}", file=sys.stderr)
            raise e

        # Validate local paths if they exist, otherwise trust HF hub names.
        for path_val, name in [(self.base_model_path, "Base model"), (self.adapter_path, "Adapter")]:
            if path_val and ("/" in path_val and not path_val.startswith(("./", "/"))):
                continue
            if path_val and not Path(path_val).exists():
                raise FileNotFoundError(f"{name} path not found: {path_val}")

        print(f"🤖 Loading HF Base Model: {self.base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            if self.adapter_path:
                print(f"🔌 Loading LoRA Adapter: {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                self.model = base_model
        except Exception as e:
            print(f"❌ Failed to load HF model/adapter: {e}", file=sys.stderr)
            raise e

    def generate(self, messages, max_tokens=2048, temperature=0.7, top_p=0.95):
        import torch
        try:
            torch.seed()

            encodings = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            encodings = {k: v.to(self.model.device) for k, v in encodings.items()}

            do_sample = temperature > 0.0

            with torch.no_grad():
                outputs = self.model.generate(
                    **encodings,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    top_k=40 if do_sample else None,
                    repetition_penalty=1.1,
                )

            prompt_len = encodings["input_ids"].shape[1]
            gen_tokens = outputs[0][prompt_len:]

            if len(gen_tokens) >= max_tokens:
                print(f"  ⚠️  WARNING: generation stopped at max_tokens limit ({max_tokens}).")
            return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"❌ Generation error: {e}", file=sys.stderr)
            raise e
