import os
import asyncio
import logging
from contextlib import contextmanager
from threading import Lock

import torch
import easyocr
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.text_engine import optimize_code_snippet

logger = logging.getLogger("devai.image_engine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_MODEL_ID = "vikhyatk/moondream2"
MAX_IMAGE_MB = 15
MIN_OCR_CONFIDENCE = 0.4

_ocr_reader = None
_vision_model = None
_vision_tokenizer = None
_load_lock = Lock()


@contextmanager
def _scoped_getattr_patch():
    """
    moondream2's custom modeling code probes for an attribute that newer
    `transformers` versions don't set on every nn.Module. The original
    patch replaced torch.nn.Module.__getattr__ globally and for the
    process's whole lifetime — that silently changes behavior for every
    model you ever load in this process, including the text engine's
    model, and makes real AttributeErrors elsewhere impossible to trust.
    This scopes the patch to just the moondream2 load call, then restores
    the original immediately after.
    """
    original = torch.nn.Module.__getattr__

    def patched(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return original(self, name)

    torch.nn.Module.__getattr__ = patched
    try:
        yield
    finally:
        torch.nn.Module.__getattr__ = original


def _load_models() -> None:
    global _ocr_reader, _vision_model, _vision_tokenizer
    if _ocr_reader is not None and _vision_model is not None:
        return
    with _load_lock:
        if _ocr_reader is not None and _vision_model is not None:
            return

        logger.info("Loading EasyOCR...")
        try:
            _ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
        except Exception:
            logger.exception("EasyOCR failed to load")
            _ocr_reader = None

        logger.info("Loading vision model (%s)...", VISION_MODEL_ID)
        try:
            _vision_tokenizer = AutoTokenizer.from_pretrained(VISION_MODEL_ID)
            with _scoped_getattr_patch():
                _vision_model = AutoModelForCausalLM.from_pretrained(
                    VISION_MODEL_ID, trust_remote_code=True
                ).to(DEVICE)
            _vision_model.eval()
            logger.info("Vision model ready on %s", DEVICE)
        except Exception:
            logger.exception("Vision model failed to load")
            _vision_model = None


def _validate_image(image_path: str) -> str | None:
    """Returns an error message, or None if the file is a usable image."""
    if not os.path.exists(image_path):
        return "Image file not found."
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        return f"Image exceeds {MAX_IMAGE_MB}MB limit ({size_mb:.1f}MB)."
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        return "File is not a valid image."
    return None


def _detect_code_patterns(text: str) -> bool:
    """Heuristic check for programming syntax in OCR output."""
    if not text.strip():
        return False
    code_keywords = [
        "def ", "class ", "import ", "function ", "const ", "let ", "var ",
        "#include", "SELECT *", "package ", "return ",
    ]
    syntax_indicators = ["{", "}", "()", "=>", "->", "::"]
    score = sum(1 for kw in code_keywords if kw in text)
    score += sum(1 for syn in syntax_indicators if syn in text)
    # A lone colon is too common in prose to count alone.
    if ":" in text and score > 0:
        score += 1
    return score >= 2


def _run_ocr(image_path: str) -> str:
    if _ocr_reader is None:
        return ""
    try:
        results = _ocr_reader.readtext(image_path, detail=1)
        kept = [text for (_, text, conf) in results if conf >= MIN_OCR_CONFIDENCE]
        return "\n".join(kept)
    except Exception:
        logger.exception("OCR extraction failed")
        return ""


def _run_vision_qa(image_path: str, user_query: str) -> str:
    if _vision_model is None:
        return "Vision model offline."
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            enc_image = _vision_model.encode_image(image)
            return _vision_model.answer_question(enc_image, user_query, _vision_tokenizer)
    except Exception as exc:
        logger.exception("Vision Q&A failed")
        return f"Error during Vision Q&A: {exc}"


async def process_image_input(
    image_path: str,
    user_query: str = "What can you see in this image?",
) -> dict:
    """Async entry point — `await` this directly from a FastAPI route.
    OCR runs first; if the extracted text looks like code, it's handed
    to the text engine for optimization, otherwise it's routed to the
    vision model for interactive Q&A. Blocking calls run in worker
    threads so the event loop stays free for other requests."""
    error = _validate_image(image_path)
    if error:
        return {"status": "error", "message": error}

    await asyncio.to_thread(_load_models)

    extracted_text = await asyncio.to_thread(_run_ocr, image_path)

    if _detect_code_patterns(extracted_text):
        logger.info("Code detected — routing to text engine")
        prompt = f"Please analyze, refactor, and explain the following code snippet:\n\n{extracted_text}"
        optimization_result = await optimize_code_snippet(prompt)

        if optimization_result["status"] == "error":
            return {
                "status": "error",
                "routed_to": "Code Optimizer Engine (via OCR)",
                "extracted_content": extracted_text,
                "message": optimization_result["message"],
            }

        return {
            "status": "success",
            "routed_to": "Code Optimizer Engine (via OCR)",
            "extracted_content": extracted_text,
            "interaction_type": "code_optimization",
            "result": {
                "raw_ocr": extracted_text,
                "optimized_output": optimization_result["optimized_output"],
            },
        }

    logger.info("No code detected — routing to vision Q&A")
    answer = await asyncio.to_thread(_run_vision_qa, image_path, user_query)
    return {
        "status": "success",
        "routed_to": "Interactive Vision Q&A Engine",
        "extracted_content": extracted_text,
        "interaction_type": "conversational_qa",
        "result": {"question": user_query, "answer": answer},
    }