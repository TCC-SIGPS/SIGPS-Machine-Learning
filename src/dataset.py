import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.path import CAMINHO_COHORT_CSV, CAMINHO_COHORT_META
from .features import ORDEM_FEATURES


def carregar_metadados() -> dict:
    if CAMINHO_COHORT_META.exists():
        return json.loads(CAMINHO_COHORT_META.read_text(encoding="utf-8"))
    return {}


def carregar_dataset(caminho: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Carrega cohort demo SIGPS (1000 pacientes) exportado do banco."""
    csv_path = caminho or CAMINHO_COHORT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {csv_path}. "
            "Rode o seed no backend e exporte o CSV, ou use python -m src.train com dados sintéticos."
        )
    df = pd.read_csv(csv_path)
    X = df[ORDEM_FEATURES].to_numpy(dtype=np.float32)
    y = df["prioridade"].to_numpy(dtype=np.int32)
    return X, y


def criar_dataset(n_samples: int = 1000):
    """Dataset sintético (fallback quando não há CSV do cohort)."""
    np.random.seed(42)

    idades = np.random.randint(18, 90, n_samples)
    diabetes = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    hipertensao = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    cancer = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    org_ids = np.random.randint(1, 6, n_samples)

    prioridades = []
    for i in range(n_samples):
        idade = idades[i]
        d = diabetes[i]
        h = hipertensao[i]
        c = cancer[i]

        if c == 1 or (idade >= 60 and (d == 1 or h == 1)) or (idade > 50 and d == 1 and h == 1):
            prioridade = 3
        elif idade >= 60 or d == 1 or h == 1:
            prioridade = 2
        else:
            prioridade = 1

        prioridades.append(prioridade)

    df = pd.DataFrame(
        {
            "idade": idades,
            "tem_diabetes": diabetes,
            "tem_hipertensao": hipertensao,
            "tem_cancer": cancer,
            "organization_id": org_ids,
            "prioridade": prioridades,
        }
    )

    X = df[ORDEM_FEATURES].to_numpy(dtype=np.float32)
    y = df["prioridade"].to_numpy(dtype=np.int32)
    return X, y
