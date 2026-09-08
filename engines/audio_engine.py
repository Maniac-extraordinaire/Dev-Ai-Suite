import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import whisper
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper and SentenceTransformer models into memory...")
whisper_model = whisper.load_model("base", device=device)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 1. Recreate the Severity Classifier architecture
class SeverityClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Resolve paths safely relative to the backend weights folder
current_dir = os.path.dirname(os.path.abspath(__file__))
model_weights_path = os.path.join(current_dir, "..", "weights", "bug_severity_model.pt")
classes_path = os.path.join(current_dir, "..", "weights", "severity_classes.json")
issues_path = os.path.join(current_dir, "..", "weights", "existing_issues.pt")

# 3. Load trained model weights and metadata
classifier = SeverityClassifier().to(device)
classifier.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
classifier.eval()

with open(classes_path, "r") as f:
    class_map = {int(k): v for k, v in json.load(f).items()}

existing_db = torch.load(issues_path, map_location=device, weights_only=False)

# 4. Core Audio Pipeline Function
def process_voice_bug_report(audio_path: str):
    if not os.path.exists(audio_path):
        return {"error": "Audio file path not found."}

    # A. Transcribe the audio file using Whisper
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"].strip()

    # B. Generate text embedding
    emb = embedder.encode([transcript], convert_to_tensor=True).to(device)

    # C. Predict severity using your PyTorch classifier
    with torch.no_grad():
        logits = classifier(emb)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        severity = class_map[pred_idx]
        confidence_dict = {class_map[i]: round(probs[i].item(), 3) for i in range(len(class_map))}

    # D. Find duplicate issues via cosine similarity against your database
    db_embs = existing_db["embeddings"].to(device)
    sims = F.cosine_similarity(emb, db_embs)
    top_k_vals, top_k_idx = torch.topk(sims, k=3)

    similar_issues = [
        (existing_db["texts"][idx], round(val.item(), 3)) 
        for val, idx in zip(top_k_vals, top_k_idx)
    ]

    return {
        "status": "success",
        "transcript": transcript,
        "predicted_severity": severity,
        "severity_confidence": confidence_dict,
        "similar_existing_issues": similar_issues,
    }
