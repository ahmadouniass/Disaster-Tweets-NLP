import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import torch

# On mock le chargement du modèle AVANT d'importer l'app
with patch("transformers.AutoModelForSequenceClassification.from_pretrained"), \
     patch("transformers.AutoTokenizer.from_pretrained"):
    from api import app

client = TestClient(app)

def test_read_root():
    """Vérifie que l'API est en ligne."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_empty_text():
    """Vérifie que l'API rejette les textes vides."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

@patch("api.model")
@patch("api.tokenizer")
def test_predict_success_mock(mock_tokenizer, mock_model):
    """Vérifie le fonctionnement de la prédiction avec un modèle simulé (Mock)."""
    # 1. Simuler la sortie du tokenizer
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    
    # 2. Simuler la sortie du modèle (Logits)
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]]) # 0.9 = Disaster
    mock_model.return_value = mock_outputs
    
    # 3. Appeler l'API
    response = client.post("/predict", json={"text": "Test disaster tweet"})
    
    # 4. Vérifications
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Disaster"
    assert data["confidence"] > 0.5

def test_invalid_json():
    """Vérifie la gestion des requêtes malformées."""
    response = client.post("/predict", content="not a json")
    assert response.status_code == 422
