# Envia artifacts/model.pkl para a VPS e reinicia o container sigps-ml.
# Uso (Windows PowerShell):
#   cd SIGPS-Machine-Learning
#   copy scripts\deploy.env.example scripts\deploy.env   # edite VPS_USER, VPS_HOST, VPS_ML_DIR
#   .\scripts\enviar_modelo_vps.ps1
#
# Opções:
#   .\scripts\enviar_modelo_vps.ps1 -Treinar   # treina localmente antes de enviar

param(
    [switch]$Treinar
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path $PSScriptRoot -Parent
$EnvFile = Join-Path $PSScriptRoot "deploy.env"
$ModeloLocal = Join-Path $Raiz "artifacts\model.pkl"

function Load-DeployEnv {
    if (-not (Test-Path $EnvFile)) {
        Write-Error "Arquivo não encontrado: $EnvFile`nCopie deploy.env.example para deploy.env e preencha VPS_USER, VPS_HOST e VPS_ML_DIR."
    }
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -match "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$") {
            Set-Variable -Name $Matches[1] -Value $Matches[2] -Scope Script
        }
    }
}

Load-DeployEnv

if ($Treinar) {
    Write-Host "Treinando modelo localmente..."
    Push-Location $Raiz
    python -m src.train
    Pop-Location
}

if (-not (Test-Path $ModeloLocal)) {
    Write-Error "Modelo não encontrado: $ModeloLocal`nRode: python -m src.train"
}

$Destino = "${VPS_USER}@${VPS_HOST}:${VPS_ML_DIR}/artifacts/model.pkl"
Write-Host "Enviando $ModeloLocal -> $Destino"
ssh "${VPS_USER}@${VPS_HOST}" "mkdir -p ${VPS_ML_DIR}/artifacts"
scp $ModeloLocal $Destino

Write-Host "Reiniciando container $ML_CONTAINER..."
ssh "${VPS_USER}@${VPS_HOST}" @"
docker restart $ML_CONTAINER
sleep 4
docker exec $ML_CONTAINER python -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())\"
"@

Write-Host "Concluído. Verifique model_loaded: true na resposta acima."
