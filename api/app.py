import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
import sys
import os

# Add src to path so we can import our model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_fusion import MindScopeFusionModel

# Initialize App
app = FastAPI(title="MindScope AI Risk API", version="1.0")

# Load Resources (Global to avoid reloading on every request)
DEVICE = "cpu" # Use CPU for inference API to save costs
MODEL_PATH = "models/mindscope_best.pth"
TOKENIZER_NAME = 'distilbert-base-uncased'

print("Loading MindScope Model...")
# Re-initialize the architecture
model = MindScopeFusionModel(num_tabular_features=4, num_classes=3)
# Load weights if they exist
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("--> Weights loaded successfully.")
else:
    print("--> WARNING: No model weights found. Using random weights (Debug Mode).")
model.to(DEVICE)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)

# Define Input Schema (Data Contract)
class PatientInput(BaseModel):
    clinical_note: str
    age: int
    sleep_hours: float
    stress_level: int
    family_history: int

@app.post("/predict_risk")
def predict_risk(patient: PatientInput):
    """
    Endpoint to assess mental health risk.
    """
    try:
        # 1. Preprocess Text
        encoding = tokenizer.encode_plus(
            patient.clinical_note,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # 2. Preprocess Tabular (Normalize as done in training)
        # Age / 100, Sleep / 24, Stress / 10
        tabular_features = torch.tensor([
            patient.age / 100.0,
            patient.sleep_hours / 24.0,
            patient.stress_level / 10.0,
            patient.family_history
        ], dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dim

        # 3. Inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask, tabular_features)
            probs = torch.softmax(logits, dim=1)
            risk_score = torch.argmax(probs, dim=1).item()
            
        # Map output to label
        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        
        return {
            "risk_category": risk_map[risk_score],
            "confidence_scores": {
                "low": f"{probs[0][0]:.4f}",
                "medium": f"{probs[0][1]:.4f}",
                "high": f"{probs[0][2]:.4f}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
