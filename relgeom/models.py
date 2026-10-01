"""Load a Hugging Face causal LM, locate its decoder blocks, and format prompts.

Two prompt formats are used in the paper:
  chat   the model's native chat template with one user message (instruction-tuned models);
  plain  [BOS] + body + "\\nAnswer:", answers continue after a space and end at the first newline
         (base weights, and instruct weights in the base-vs-instruct comparison).
Computation is FP32 without TF32, as in the paper.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Main blocks (zero-based block outputs), chosen on separate block-selection word sets by the rule of
# App. A (`experiments/select_block.py`). Keys: (Hugging Face id, prompt format).
MAIN_BLOCKS = {
    ("Qwen/Qwen2.5-7B-Instruct", "chat"): 20,
    ("google/gemma-3-4b-it", "chat"): 22,
    ("meta-llama/Llama-3.1-8B-Instruct", "chat"): 15,
    # Scale sweep (Table 5)
    ("Qwen/Qwen2.5-1.5B-Instruct", "chat"): 18,
    ("Qwen/Qwen2.5-3B-Instruct", "chat"): 26,
    ("Qwen/Qwen2.5-14B-Instruct", "chat"): 34,
    ("Qwen/Qwen2.5-32B-Instruct", "chat"): 48,
    ("Qwen/Qwen3-8B", "chat"): 22,
    ("meta-llama/Llama-3.2-3B-Instruct", "chat"): 14,
    # Base versus instruction-tuned weights on the plain prompt (Table 6)
    ("Qwen/Qwen2.5-1.5B", "plain"): 18,
    ("Qwen/Qwen2.5-1.5B-Instruct", "plain"): 20,
    ("Qwen/Qwen2.5-3B", "plain"): 26,
    ("Qwen/Qwen2.5-3B-Instruct", "plain"): 26,
    ("Qwen/Qwen3-8B-Base", "plain"): 22,
    ("Qwen/Qwen3-8B", "plain"): 22,
    ("google/gemma-3-4b-pt", "plain"): 22,
    ("google/gemma-3-4b-it", "plain"): 20,
    ("meta-llama/Llama-3.2-3B", "plain"): 14,
    ("meta-llama/Llama-3.2-3B-Instruct", "plain"): 14,
    # OLMo-2 checkpoints are all read at the released model's block (App. F.2).
    ("allenai/OLMo-2-1124-7B", "plain"): 16,
}
PLAIN_CUE = "\nAnswer:"
MAX_NEW_TOKENS = 16


def main_block(model_id, prompt_format):
    return MAIN_BLOCKS.get((model_id, prompt_format))


def _get(obj, path):
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def find_decoder(model):
    """The module holding the decoder block list, across Qwen, Llama, OLMo-2 and the Gemma-3 wrapper."""
    for path in ("model.language_model", "language_model.model", "model", "gpt_neox"):
        try:
            decoder = _get(model, path)
        except AttributeError:
            continue
        if hasattr(decoder, "layers"):
            return decoder
    raise ValueError(f"Cannot locate decoder layers of {type(model).__name__}")


class LM:
    """A frozen model plus everything needed to render prompts and read or edit block outputs."""

    def __init__(self, model_id, revision=None, prompt_format="chat", dtype="float32"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.model_id, self.revision, self.prompt_format = model_id, revision, prompt_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        if config.model_type == "gemma3":
            # Gemma-3 4B ships as an image-text wrapper; only its text path is used.
            from transformers import AutoModelForImageTextToText as loader
        else:
            loader = AutoModelForCausalLM
        device_map = "auto" if torch.cuda.device_count() > 1 else {"": 0}
        self.model = loader.from_pretrained(model_id, revision=revision, torch_dtype=getattr(torch, dtype),
                                            attn_implementation="eager", device_map=device_map).eval()
        self.family = config.model_type
        decoder = find_decoder(self.model)
        self.layers = decoder.layers
        self.final_norm = getattr(decoder, "norm", None) or decoder.final_layer_norm
        self.n_blocks = len(self.layers)
        self.hidden_size = self.model.get_input_embeddings().weight.shape[1]
        self.device = self.model.get_input_embeddings().weight.device
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set([eos] if isinstance(eos, int) else (eos or [self.tokenizer.eos_token_id]))
        pad = self.model.generation_config.pad_token_id
        self.pad_id = pad if pad is not None else (self.tokenizer.pad_token_id or min(self.eos_ids))
        # Plain format: any token containing a newline ends an answer, but may not open one
        # (some base models start every continuation with a blank line).
        self.newline_ids = set()
        if prompt_format == "plain":
            pieces = self.tokenizer.batch_decode([[i] for i in range(len(self.tokenizer))])
            self.newline_ids = {i for i, piece in enumerate(pieces) if "\n" in piece} - self.eos_ids

    # ------------------------------------------------------------------ prompts
    def render(self, body):
        """Prompt text for one user message; the query state is read at its last token."""
        if self.prompt_format == "plain":
            # OLMo-2 packs pretraining documents separated by <|endoftext|>, which therefore marks a document start.
            bos = self.tokenizer.bos_token or ("<|endoftext|>" if "OLMo" in self.model_id else "")
            return bos + body + PLAIN_CUE
        kwargs = {}
        if self.model_id.startswith("Qwen/Qwen3"):
            kwargs["enable_thinking"] = False          # Qwen3 in non-thinking mode
        if "Llama-3" in self.model_id:
            kwargs["date_string"] = "26 Jul 2024"       # Llama templates otherwise insert today's date
        return self.tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                                  add_generation_prompt=True, **kwargs)

    def encode(self, body):
        return self.tokenizer(self.render(body), add_special_tokens=False)["input_ids"]

    def answer_ids(self, prompt_ids, answer):
        """Token ids of an answer continuation after the fixed prompt (no EOS)."""
        text = " " + answer if self.prompt_format == "plain" else answer
        prompt = self.tokenizer.decode(prompt_ids)
        full = self.tokenizer(prompt + text, add_special_tokens=False)["input_ids"]
        if full[:len(prompt_ids)] == list(prompt_ids):
            return full[len(prompt_ids):]
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def first_token(self, answer, prompt_ids=None):
        """First answer token; the answer header is fixed, so any prompt of the task gives the same id."""
        if prompt_ids is None:
            prompt_ids = self.encode("Answer the question.\n\nWhat is it?")
        return self.answer_ids(prompt_ids, answer)[0]

    def output_directions(self, token_ids):
        """Output direction g * W_U[t] of each token: final-norm gain times its unembedding row."""
        W = self.model.get_output_embeddings().weight
        rows = W.detach()[torch.tensor(list(token_ids), device=W.device)].float().cpu()
        gain = self.final_norm.weight.detach().float().cpu()
        if self.family.startswith("gemma"):
            gain = gain + 1.0                           # Gemma RMSNorm scales by (1 + w)
        return (rows * gain).numpy().astype("float64")
