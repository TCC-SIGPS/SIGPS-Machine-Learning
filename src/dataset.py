import numpy as np
import pandas as pd
from .features import ORDEM_FEATURES

def criar_dataset(n_samples: int = 1000):
    """
    Gera um dataset sintético para treinar o modelo de classificação de risco.
    Regras de Prioridade:
      - 3 (Extrema): Se tiver alguma comorbidade e idade >= 60, ou se tiver câncer, ou diabetes+hipertensão e idade > 50.
      - 2 (Alta): Idosos (>= 60) sem comorbidades graves, ou < 60 com diabetes ou hipertensão.
      - 1 (Normal): < 60 sem comorbidades.
    """
    np.random.seed(42)
    
    idades = np.random.randint(18, 90, n_samples)
    diabetes = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    hipertensao = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    cancer = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    prioridades = []
    
    for i in range(n_samples):
        idade = idades[i]
        d = diabetes[i]
        h = hipertensao[i]
        c = cancer[i]
        
        if c == 1 or (idade >= 60 and (d == 1 or h == 1)) or (idade > 50 and d == 1 and h == 1):
            prioridade = 3 # Extrema
        elif idade >= 60 or d == 1 or h == 1:
            prioridade = 2 # Alta
        else:
            prioridade = 1 # Normal
            
        prioridades.append(prioridade)
        
    df = pd.DataFrame({
        "idade": idades,
        "tem_diabetes": diabetes,
        "tem_hipertensao": hipertensao,
        "tem_cancer": cancer,
        "prioridade": prioridades
    })
    
    X = df[ORDEM_FEATURES].to_numpy(dtype=np.float32)
    y = df["prioridade"].to_numpy(dtype=np.int32)
    
    return X, y