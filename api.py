import logging
import os
import time

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel

from priority_rules import calcular_prioridade

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SIGPS-ML] %(levelname)s %(message)s',
)
logger = logging.getLogger('sigps-ml')

app = FastAPI(title='SIGPS ML API')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'artifacts', 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    logger.info('Modelo carregado: %s', MODEL_PATH)
except Exception as exc:
    logger.warning('Modelo não carregado (%s). Usando fallback de regras.', exc)
    model = None


class PredictionRequest(BaseModel):
    idade: int
    tem_diabetes: int
    tem_hipertensao: int
    tem_cancer: int
    organization_id: int


@app.middleware('http')
async def log_requests(request: Request, call_next):
    inicio = time.perf_counter()
    response = await call_next(request)
    duracao_ms = (time.perf_counter() - inicio) * 1000
    logger.info('%s %s -> %s (%.0f ms)', request.method, request.url.path, response.status_code, duracao_ms)
    return response


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
    }


@app.post('/predict')
def predict_priority(req: PredictionRequest):
    if model is None:
        prioridade = calcular_prioridade(
            req.idade, req.tem_diabetes, req.tem_hipertensao, req.tem_cancer
        )
        logger.info(
            'predict fallback org=%s idade=%s diabetes=%s hipertensao=%s cancer=%s -> %s',
            req.organization_id, req.idade, req.tem_diabetes, req.tem_hipertensao, req.tem_cancer, prioridade,
        )
        return {'prioridade': prioridade}

    features = [[req.idade, req.tem_diabetes, req.tem_hipertensao, req.tem_cancer, req.organization_id]]
    prioridade = int(model.predict(features)[0])
    logger.info(
        'predict modelo org=%s idade=%s diabetes=%s hipertensao=%s cancer=%s -> %s',
        req.organization_id, req.idade, req.tem_diabetes, req.tem_hipertensao, req.tem_cancer, prioridade,
    )
    return {'prioridade': prioridade}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('api:app', host='127.0.0.1', port=8000, reload=True, access_log=True)
