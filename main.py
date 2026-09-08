import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from engines.numerical_engine import process_traffic_data
from engines.audio_engine import process_voice_bug_report
from engines.text_engine import optimize_code_snippet
from engines.image_engine import process_image_input

app = FastAPI(
    title="DevAI Suite API",
    description="Production API for DevAI Suite"
)

# 2. Add CORS middleware right here to allow browser requests and OPTIONS headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    # Allows custom headers like ngrok-skip-browser-warning
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)


def _save_upload(file: UploadFile, raw_bytes: bytes) -> Path:
    """Writes an upload under a random name, keeping only the extension
    from the client-supplied filename. Two concurrent uploads called
    'image.png' used to overwrite the same `temp_{filename}` path and
    could race each other's read/delete; a crafted filename like
    '../../engines/text_engine.py' could also escape the temp dir
    entirely since the original filename went straight into the path."""
    suffix = Path(file.filename or "").suffix
    dest = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(raw_bytes)
    return dest


@app.get("/", response_class=FileResponse)
def home():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "static", "index.html")


@app.post("/api/predict-traffic")
async def api_predict_traffic(
    source: str = Form(...),
    pasted_string: str = Form(None)
):
    data = pasted_string if source == "string" else None
    return process_traffic_data(source=source, data=data)


@app.post("/api/triage-bug")
async def api_triage_bug(file: UploadFile = File(...)):
    temp_path = _save_upload(file, await file.read())
    try:
        result = process_voice_bug_report(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)
    return result


@app.post("/api/optimize-code")
async def api_optimize_code(prompt: str = Form(...)):
    # optimize_code_snippet is now async (runs generation in a worker
    # thread) — it must be awaited, or FastAPI serializes a coroutine
    # object instead of the actual result.
    return await optimize_code_snippet(prompt)


@app.post("/api/process-image")
async def api_process_image(
    file: UploadFile = File(...),
    query: str = Form("Analyze this image.")
):
    temp_path = _save_upload(file, await file.read())
    try:
        # process_image_input is likewise now async.
        result = await process_image_input(str(temp_path), user_query=query)
    finally:
        temp_path.unlink(missing_ok=True)
    return result
