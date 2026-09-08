# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# 1. Hugging Face Spaces strictly requires a non-root user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 2. Temporarily switch to root to install core Linux dependencies
USER root
# Install ffmpeg (required for Audio Engine) and libGL (required for EasyOCR)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Switch back to our secure user
USER user

# 4. Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the DevAI Suite into the container
COPY --chown=user . .

# 6. Hugging Face Spaces strictly listens on port 7860
EXPOSE 7860

# 7. Boot the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]