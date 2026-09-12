# 🚀 DevAI Suite

A production-grade, multi-modal machine learning platform featuring a decoupled client-server architecture. DevAI Suite bridges a lightweight local frontend with a high-performance cloud GPU backend to process complex, cross-domain AI workloads in real time.

## 🏗️ System Architecture

Unlike monolithic data-science dashboards (e.g., Streamlit), this project mimics a real-world production environment by separating the UI from the heavy GPU inference engine. 

*   **Frontend:** Vanilla HTML/JS/CSS web client processing asynchronous API requests.
*   **Networking:** Secure public tunneling via Ngrok with custom CORS middleware to handle browser preflight (`OPTIONS`) constraints.
*   **Backend:** FastAPI microservice hosted on Google Colab, leveraging an NVIDIA T4 GPU for high-speed tensor operations.
*   **Storage:** Heavy model weights ($>2$GB) are decoupled from version control and mounted dynamically at runtime via Google Drive.

---

## ⚡ Core Engines & Features

### 👁️ Vision & OCR (Moondream2)
Upload screenshots of code, system architecture diagrams, or UI errors. The engine performs OCR and leverages a compact vision-language model to answer questions interactively or route extracted text to the code optimizer.

### 💻 Code Optimizer
A fine-tuned text adapter designed to refactor inefficient code. Features include $O(1)$ complexity optimizations, injecting safe context managers, and adding robust exception handling. 

### 🎙️ Audio Bug Triage (Whisper + PyTorch)
Upload audio files or use browser microphone recording to describe a software bug. The engine transcribes the audio using OpenAI's Whisper and passes the text through a custom PyTorch classifier to predict severity (e.g., Critical, High, Low) and surface duplicate tickets.

### 📈 Numerical Traffic Forecasting (LSTM)
Predicts next-hour traffic conditions using a trained PyTorch Long Short-Term Memory (LSTM) network, capable of ingesting live mock API streams or manual comma-separated values.

---

## 🛠️ Tech Stack

*   **Backend Framework:** FastAPI, Uvicorn, Python
*   **Machine Learning:** PyTorch, HuggingFace Transformers, Moondream2, OpenAI Whisper, Torchvision
*   **Networking & Integration:** Ngrok, RESTful APIs, CORS Middleware
*   **Frontend:** HTML5, CSS3, ES6 JavaScript, Fetch API

---

## 🚀 How to Run the Project

### 1. Launch the Cloud Backend (GPU)
1. Open the provided Jupyter Notebook (`DevAI_Backend.ipynb`) in Google Colab.
2. Ensure the runtime is set to **T4 GPU** (`Runtime > Change runtime type`).
3. Add your Ngrok Auth Token in the designated cell.
4. Run all cells. The script will automatically clone this repository, install pre-built binary dependencies to optimize disk space, mount model weights, and spin up the FastAPI server.
5. Copy the generated public Ngrok URL from the terminal output.

### 2. Connect the Local Client
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/Maniac-extraordinaire/Dev-Ai-Suite.git](https://github.com/Maniac-extraordinaire/Dev-Ai-Suite.git)
