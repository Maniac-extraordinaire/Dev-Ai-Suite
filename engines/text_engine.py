import os
import asyncio
import logging
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("devai.text_engine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "text_adapter")
)

_model = None
_tokenizer = None
_load_lock = Lock()


def _load_model() -> None:
    """Loads the fine-tuned text model once, lazily, thread-safe against
    concurrent FastAPI requests racing to trigger the first load."""
    global _model, _tokenizer
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:  # re-check inside the lock
            return
        logger.info("Loading text model from %s", _MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_DIR,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        _model.eval()
        logger.info("Text model ready on %s", DEVICE)


def _generate_sync(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    _load_model()
    formatted_prompt = (
        "System: You are an expert AI software engineer. Provide clear, structured, "
        "and professional explanations with proper markdown formatting, bullet points, "
        "and optimized code blocks.\n\n"
        f"User Request: {prompt}\n\nAssistant:"
    )
    inputs = _tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    return decoded[len(formatted_prompt):].strip()


async def optimize_code_snippet(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> dict:
    """Async entry point — `await` this directly from a FastAPI route.
    Runs the blocking generate() call in a worker thread so it never
    stalls the event loop."""
    try:
        cleaned_output = await asyncio.to_thread(
            _generate_sync, prompt, max_tokens, temperature, top_p
        )
        return {"status": "success", "input_prompt": prompt, "optimized_output": cleaned_output}
    except Exception as exc:
        logger.exception("Text generation failed")
        return {"status": "error", "input_prompt": prompt, "message": str(exc)}