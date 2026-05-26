import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from core.path import CAMINHO_MODELO, CAMINHO_METRICAS, PASTA_ARTIFACTS
from .dataset import criar_dataset
from .features import ORDEM_FEATURES

def main():
    # 1) cria pasta artifacts/
    PASTA_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 2) carrega dados (gera sintético)
    print("Gerando dataset sintético...")
    X, y = criar_dataset(n_samples=2000)

    # 3) split treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    # 5) treino
    print("Treinando modelo RandomForestClassifier...")
    modelo.fit(X_treino, y_treino)

    # 6) predição
    pred = modelo.predict(X_teste)

    # 7) métricas
    acc = float(accuracy_score(y_teste, pred))

    # 8) salva modelo
    joblib.dump(modelo, CAMINHO_MODELO)

    # 9) salva métricas + contrato
    payload = {
        "accuracy": acc,
        "ordem_features": ORDEM_FEATURES,
        "n_treino": int(len(X_treino)),
        "n_teste": int(len(X_teste)),
        "modelo": type(modelo).__name__,
    }
    CAMINHO_METRICAS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Treino concluido")
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Métricas salvas em: {CAMINHO_METRICAS}")
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()