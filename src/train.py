import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from core.path import CAMINHO_METRICAS, CAMINHO_MODELO, PASTA_ARTIFACTS
from .dataset import carregar_dataset, carregar_metadados
from .features import ORDEM_FEATURES

CLASSES = [1, 2, 3]


def main():
    PASTA_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    meta = carregar_metadados()
    print(f"Carregando dataset: {meta.get('nome', 'SIGPS demo 1000 pacientes')}...")
    X, y = carregar_dataset()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Treinando RandomForestClassifier...")
    modelo.fit(X_treino, y_treino)

    pred = modelo.predict(X_teste)
    acc = float(accuracy_score(y_teste, pred))
    f1_macro = float(f1_score(y_teste, pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_teste, pred, average="weighted", zero_division=0))
    recall_extrema = float(recall_score(y_teste, pred, labels=[3], average="macro", zero_division=0))

    cm = confusion_matrix(y_teste, pred, labels=CLASSES).tolist()
    report = classification_report(
        y_teste,
        pred,
        labels=CLASSES,
        target_names=["Normal", "Alta", "Extrema"],
        output_dict=True,
        zero_division=0,
    )

    joblib.dump(modelo, CAMINHO_MODELO)

    payload = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "recall_extrema": round(recall_extrema, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "ordem_features": ORDEM_FEATURES,
        "n_treino": int(len(X_treino)),
        "n_teste": int(len(X_teste)),
        "n_total": int(len(y)),
        "modelo": type(modelo).__name__,
        "classes": CLASSES,
        "dataset": meta,
    }
    CAMINHO_METRICAS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Treino concluido")
    print(f"Modelo: {CAMINHO_MODELO}")
    print(f"Metricas: {CAMINHO_METRICAS}")
    print(f"Accuracy: {acc:.4f} | F1 macro: {f1_macro:.4f} | Recall Extrema: {recall_extrema:.4f}")


if __name__ == "__main__":
    main()
