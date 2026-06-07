#!/usr/bin/env bash
# Envia artifacts/model.pkl para a VPS e reinicia o container sigps-ml.
#
# Uso:
#   cd SIGPS-Machine-Learning
#   cp scripts/deploy.env.example scripts/deploy.env   # edite as variáveis
#   bash scripts/enviar_modelo_vps.sh
#   bash scripts/enviar_modelo_vps.sh --treinar

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/scripts/deploy.env"
MODEL_LOCAL="$ROOT/artifacts/model.pkl"
TREINAR=false

for arg in "$@"; do
  case "$arg" in
    --treinar) TREINAR=true ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Crie scripts/deploy.env a partir de deploy.env.example"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${VPS_USER:?defina VPS_USER em deploy.env}"
: "${VPS_HOST:?defina VPS_HOST em deploy.env}"
: "${VPS_ML_DIR:?defina VPS_ML_DIR em deploy.env}"
: "${ML_CONTAINER:=sigps-ml}"

if [[ "$TREINAR" == true ]]; then
  echo "Treinando modelo localmente..."
  (cd "$ROOT" && python -m src.train)
fi

if [[ ! -f "$MODEL_LOCAL" ]]; then
  echo "Modelo não encontrado: $MODEL_LOCAL"
  echo "Rode: python -m src.train"
  exit 1
fi

DEST="${VPS_USER}@${VPS_HOST}:${VPS_ML_DIR}/artifacts/model.pkl"
echo "Enviando $MODEL_LOCAL -> $DEST"
ssh "${VPS_USER}@${VPS_HOST}" "mkdir -p ${VPS_ML_DIR}/artifacts"
scp "$MODEL_LOCAL" "$DEST"

echo "Reiniciando container ${ML_CONTAINER}..."
ssh "${VPS_USER}@${VPS_HOST}" "
  docker restart ${ML_CONTAINER}
  sleep 4
  docker exec ${ML_CONTAINER} python -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())\"
"

echo "Concluído. Verifique model_loaded: true na resposta acima."
