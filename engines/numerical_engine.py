import torch
import torch.nn as nn
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Recreate the exact LSTM architecture
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. Correctly resolve the path relative to the backend folder
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, "..", "weights", "traffic_lstm.pt")

traffic_model = TrafficLSTM().to(device)

# 3. Safely load handling both raw weights and checkpoint dictionaries
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    traffic_model.load_state_dict(checkpoint["model_state_dict"])
else:
    traffic_model.load_state_dict(checkpoint)

traffic_model.eval()

# 4. The prediction logic for all 3 input methods
def process_traffic_data(source: str, data=None):
    sequence = []
    
    # Method A: Mock Live API ("Wow Factor")
    if source == "api":
        time_steps = np.linspace(0, 4 * np.pi, 144)
        sequence = (np.sin(time_steps) * 50 + 150 + np.random.normal(0, 4, 144)).tolist()
        
    # Method B: CSV Upload
    elif source == "csv" and data:
        for row in data:
            val = row[0] if isinstance(row, list) else row
            try: 
                sequence.append(float(val))
            except (ValueError, TypeError): 
                continue
            
    # Method C: Pasted text
    elif source == "string" and data:
        for item in str(data).split(","):
            try: 
                sequence.append(float(item.strip()))
            except ValueError: 
                continue

    if len(sequence) < 144:
        return {"error": f"Need at least 144 data points. Only found {len(sequence)}."}
        
    # Format for PyTorch (Batch=1, Seq_Len=144, Features=1)
    input_array = np.array(sequence[-144:], dtype=np.float32)
    tensor_input = torch.tensor(input_array).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        prediction = traffic_model(tensor_input).item()

    return {
        "status": "success",
        "input_method": source,
        "current_traffic": round(float(sequence[-1]), 2),
        "predicted_next_hour": round(float(prediction), 2)
    }
