# SIGPS Machine Learning API

Repositório dedicado ao treinamento do modelo de Inteligência Artificial e à API de inferência do SIGPS, projetada para a classificação de prioridade de pacientes em filas de espera.

## Objetivo

- Treinar o modelo de Machine Learning (`RandomForestClassifier`) usando os dados do paciente (Idade, Comorbidades, Alergias, etc.).
- Expor o modelo treinado através de uma **API RESTful (FastAPI)** rodando na porta 8000.
- Classificar a prioridade clínica em Categorias (1: Normal, 2: Alta, 3: Extrema).
- Interagir com o Backend Flask do SIGPS de forma completamente autônoma.

## Setup e Instalação (Local ou VPS)

Este serviço foi projetado para rodar isoladamente do Backend principal, consumindo as dependências necessárias para Data Science.

```bash
# 1. Ative o ambiente virtual (pode usar o mesmo do Backend ou criar um novo)
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 2. Instale as dependências
pip install -r requirements.txt
```

## Como rodar o serviço

A arquitetura atual exige que a API de ML fique executando em segundo plano.

### Treinamento Inicial (Obrigatório)
Antes de iniciar a API pela primeira vez, você **deve treinar o modelo** para que o arquivo `src/model.pkl` seja gerado.

```bash
python src/train.py
```

### Iniciando a API Localmente
```bash
python -m uvicorn api:app --reload --port 8000
```
> O serviço ficará escutando requisições em `http://127.0.0.1:8000`.

### Rodando em Produção (VPS Linux)
Para manter o serviço no ar de forma robusta, recomendamos o uso de PM2 ou Systemd:

**Exemplo usando PM2:**
```bash
pm2 start "python -m uvicorn api:app --host 0.0.0.0 --port 8000" --name "sigps-ml"
pm2 save
```

## Estrutura do repositório

```
src/
├── features.py    – Engenharia de atributos (mapeamento de comorbidades para One-Hot).
├── train.py       – Rotina de treinamento (RandomForest, salvamento do pickle).
api.py             – Servidor FastAPI que expõe o endpoint /predict.
requirements.txt   – Dependências (FastAPI, Scikit-learn, Uvicorn).
README.md          – Este guia.
```

## Integração Automática com o Backend

Na arquitetura antiga, o Backend Flask lia o arquivo `model.pkl` diretamente. Na **nova arquitetura (Microserviço)**, o processo de comunicação acontece via HTTP:

1. O Paciente preenche sua "Ficha Médica" no Frontend informando `comorbidades` e `alergias`.
2. O Paciente faz Check-in na fila de espera.
3. O Backend Flask envia automaticamente um `POST http://127.0.0.1:8000/predict` contendo o payload do paciente em formato JSON:
   ```json
   {
       "idade": 65,
       "comorbidades": "Diabetes, Hipertensão",
       "alergias": "Nenhuma"
   }
   ```
4. A API FastAPI recebe a requisição, formata os dados usando o `features.py`, submete ao modelo carregado em memória, e devolve a resposta:
   ```json
   {
       "prioridade": 3,
       "reasoning": "Idoso (>=65 anos) com fatores de risco detectados."
   }
   ```
5. O Backend Flask salva essas informações na tabela `fila_atendimento` e a Fila é reordenada na mesma hora para todos os usuários da clínica.

### Observações Finais
- As comorbidades informadas pelos pacientes são transformadas de texto livre para _features binárias_ utilizando algoritmos de NLP básicos dentro do `features.py`.
- O paciente pode enviar documentos médicos (Laudos) via portal para validar juridicamente as comorbidades inseridas, mantendo a confiabilidade da IA.
