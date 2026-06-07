# SIGPS Machine Learning — Docker

| Item | Valor |
|------|--------|
| **Imagem** | `sigps-ml` |
| **Container** | `sigps-ml` |
| **Porta** | `8000` (rede interna Docker) |

## Subir em produção

```bash
cd SIGPS-Machine-Learning
# Garanta que artifacts/model.pkl existe (python -m src.train)
docker compose -f docker-compose.prod.yml up -d --build
```

O `docker-compose.prod.yml` monta `./artifacts/model.pkl` no container. **Para atualizar o modelo sem rebuild**, copie o arquivo na VPS e reinicie:

```bash
docker restart sigps-ml
```

Ou use o script local: `scripts/enviar_modelo_vps.ps1` / `enviar_modelo_vps.sh` (ver README §13).

## Build da imagem

```bash
docker build -t sigps-ml:latest .
```

Copia `api.py`, `priority_rules.py` e `artifacts/` (incluindo `model.pkl` no build inicial).

## Health

```bash
docker exec sigps-ml python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read())"
```

Resposta esperada: `"model_loaded": true`

O backend acessa via rede Docker: `http://sigps-ml:8000/predict`
