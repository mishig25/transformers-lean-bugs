"""Real `transformers serve` stack (ModelManager + GenerationState + ChatCompletionHandler) around a tiny local
Qwen2 checkpoint on CPU. Used by the reproducers in this folder."""

import tempfile

from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.cli.serving.chat_completion import ChatCompletionHandler
from transformers.cli.serving.model_manager import ModelManager
from transformers.cli.serving.utils import GenerationState


def make_tiny_repo() -> str:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1)
    repo = tempfile.mkdtemp()
    Qwen2ForCausalLM(cfg).eval().save_pretrained(repo)
    tok.save_pretrained(repo)
    return repo


def make_stack(model_timeout: int, continuous_batching: bool, cb_config=None):
    mm = ModelManager(device="cpu", dtype="float32", model_timeout=model_timeout,
                      continuous_batching=continuous_batching)
    state = GenerationState(continuous_batching=continuous_batching, cb_config=cb_config)
    handler = ChatCompletionHandler(model_manager=mm, generation_state=state)
    return mm, state, handler


def body(repo: str, stream: bool, max_tokens: int = 4) -> dict:
    return {"model": repo, "messages": [{"role": "user", "content": "hi"}], "stream": stream,
            "max_tokens": max_tokens, "temperature": 0.0}


_REPO = None


def make_repo_once() -> str:
    global _REPO
    if _REPO is None:
        _REPO = make_tiny_repo()
    return _REPO
