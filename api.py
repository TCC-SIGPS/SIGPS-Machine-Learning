from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="SIGPS ML API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Aviso: Não foi possível carregar o modelo em {MODEL_PATH}")
    model = None

class PredictionRequest(BaseModel):
    idade: int
    tem_diabetes: int
    tem_hipertensao: int
    tem_cancer: int

@app.post("/predict")
def predict_priority(req: PredictionRequest):
    if model is None:
        # Fallback
        if req.tem_cancer == 1 or (req.idade >= 60 and (req.tem_diabetes == 1 or req.tem_hipertensao == 1)):
            return {"prioridade": 3}
        elif req.idade >= 60 or req.tem_diabetes == 1 or req.tem_hipertensao == 1:
            return {"prioridade": 2}
        return {"prioridade": 1}
        
    features = [[req.idade, req.tem_diabetes, req.tem_hipertensao, req.tem_cancer]]
    pred = model.predict(features)[0]
    return {"prioridade": int(pred)}

# Executar com: uvicorn api:app --reload --port 8000
