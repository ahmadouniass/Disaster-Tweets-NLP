from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

app = FastAPI(title="Disaster Tweet Prediction API")

# Path to the exported model
MODEL_PATH = "./models/distilbert_disaster"

# Load model and tokenizer
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class TweetRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TweetRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text")
    
    try:
        # Tokenization
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # DistilBERT doesn't use token_type_ids, we must remove it if present
        inputs.pop("token_type_ids", None)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, prediction_idx = torch.max(probs, dim=1)
            
        label = "Disaster" if prediction_idx.item() == 1 else "No Disaster"
        
        return {
            "prediction": label,
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Disaster Tweet Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
