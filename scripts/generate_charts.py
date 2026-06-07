"""Gera PNGs e métricas de avaliação no contexto SIGPS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PASTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_RAIZ))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.dataset import carregar_dataset, carregar_metadados
from src.features import ORDEM_FEATURES
from src.priority_rules import calcular_prioridade

ARTIFACTS = PASTA_RAIZ / "artifacts"
CLASSES = [1, 2, 3]
NOMES = {1: "Normal", 2: "Alta", 3: "Extrema"}
NOMES_COMPLETOS = {1: "Normal (1)", 2: "Alta (2)", 3: "Extrema (3)"}
NOTA_AVALIACAO = (
    "Comparação com as regras de triagem SIGPS (idade + diabetes + hipertensão + câncer). "
    "100% = a IA repetiu as regras; não significa validação clínica em hospital."
)

CASOS_SIGPS = [
    {
        "codigo": "P1",
        "nome": "Lucas",
        "descricao": "28 anos · sem comorbidade",
        "idade": 28,
        "tem_diabetes": 0,
        "tem_hipertensao": 0,
        "tem_cancer": 0,
        "esperado": 1,
        "nivel_label": "Normal (1)",
    },
    {
        "codigo": "P2",
        "nome": "Mariana",
        "descricao": "45 anos · diabetes",
        "idade": 45,
        "tem_diabetes": 1,
        "tem_hipertensao": 0,
        "tem_cancer": 0,
        "esperado": 2,
        "nivel_label": "Alta (2)",
    },
    {
        "codigo": "P3",
        "nome": "José",
        "descricao": "72 anos · diabetes + hipertensão",
        "idade": 72,
        "tem_diabetes": 1,
        "tem_hipertensao": 1,
        "tem_cancer": 0,
        "esperado": 3,
        "nivel_label": "Extrema (3)",
    },
    {
        "codigo": "P4",
        "nome": "Ana",
        "descricao": "55 anos · câncer",
        "idade": 55,
        "tem_diabetes": 0,
        "tem_hipertensao": 0,
        "tem_cancer": 1,
        "esperado": 3,
        "nivel_label": "Extrema (3)",
    },
    {
        "codigo": "P5",
        "nome": "Carlos",
        "descricao": "65 anos · sem comorbidade",
        "idade": 65,
        "tem_diabetes": 0,
        "tem_hipertensao": 0,
        "tem_cancer": 0,
        "esperado": 2,
        "nivel_label": "Alta (2)",
    },
    {
        "codigo": "P6",
        "nome": "Caso extra (regra SIGPS)",
        "descricao": "52 anos · diabetes + hipertensão",
        "idade": 52,
        "tem_diabetes": 1,
        "tem_hipertensao": 1,
        "tem_cancer": 0,
        "esperado": 3,
        "nivel_label": "Extrema (3)",
    },
]

NOTA_VALIDACAO_DEMO = (
    "Seis perfis clínicos de exemplo (cinco pacientes demo + um caso da regra >50 anos com diabetes e hipertensão). "
    "Barra verde = IA acertou a prioridade esperada (Normal, Alta ou Extrema)."
)


def _figfootnote(fig: plt.Figure, texto: str = NOTA_AVALIACAO) -> None:
    fig.text(0.5, 0.01, texto, ha="center", fontsize=8, color="#555555", wrap=True)


def _plotar_matriz_confusao(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    n_teste: int,
    n_total: int,
    acuracia: float,
    destino: Path,
) -> None:
    """Heatmap clássico com anotações por célula."""
    rotulos = [NOMES[c] for c in CLASSES]
    anotacoes = np.array(
        [[f"{int(cm[i, j])}\n({cm_norm[i, j]:.0%})" for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm,
        annot=anotacoes,
        fmt="",
        cmap="YlGnBu",
        linewidths=1.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Nº de pacientes", "shrink": 0.85},
        xticklabels=rotulos,
        yticklabels=rotulos,
        ax=ax,
        vmin=0,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Predito pelo Random Forest", fontsize=11, labelpad=10)
    ax.set_ylabel("Verdadeiro (regras SIGPS)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Matriz de confusão\n"
        f"Amostra de teste: {n_teste} pacientes (20% de {n_total}) · Acurácia: {acuracia:.1%}",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=10, rotation=0)

    _figfootnote(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(destino, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _gerar_validacao_casos_sigps(modelo) -> dict:
    linhas = []
    for caso in CASOS_SIGPS:
        feats = [
            caso["idade"],
            caso["tem_diabetes"],
            caso["tem_hipertensao"],
            caso["tem_cancer"],
            1,
        ]
        pred_ml = int(modelo.predict([feats])[0])
        pred_regra = calcular_prioridade(
            caso["idade"], caso["tem_diabetes"], caso["tem_hipertensao"], caso["tem_cancer"]
        )
        linhas.append(
            {
                **caso,
                "predito_ml": pred_ml,
                "predito_regras": pred_regra,
                "acerto_ml": pred_ml == caso["esperado"],
                "acerto_regras": pred_regra == caso["esperado"],
            }
        )

    df = pd.DataFrame(linhas)
    rotulos_y = [f"{row['nome']}\n{row['descricao']}" for _, row in df.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = ["#2ecc71" if ok else "#e74c3c" for ok in df["acerto_ml"]]
    ax.barh(rotulos_y, [1] * len(df), color=cores)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    for i, row in df.iterrows():
        ax.text(
            0.02,
            i,
            f"IA previu: {row['predito_ml']} ({NOMES[row['predito_ml']]}) · "
            f"Esperado: {row['esperado']} ({NOMES[row['esperado']]})",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax.set_title("Validação — pacientes de exemplo da clínica demo SIGPS")
    ax.invert_yaxis()
    _figfootnote(fig, NOTA_VALIDACAO_DEMO)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(ARTIFACTS / "validacao_casos_sigps.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "n_casos": len(linhas),
        "acertos_ml": int(df["acerto_ml"].sum()),
        "acertos_regras": int(df["acerto_regras"].sum()),
        "casos": linhas,
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["font.size"] = 11

    modelo_path = ARTIFACTS / "model.pkl"
    if not modelo_path.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode: python -m src.train")

    X, y = carregar_dataset()
    _, X_teste, _, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = joblib.load(modelo_path)
    y_pred = modelo.predict(X_teste)
    y_proba = modelo.predict_proba(X_teste)

    y_regras = np.array(
        [
            calcular_prioridade(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            for r in X_teste
        ]
    )
    fidelidade_regras = float((y_pred == y_regras).mean())

    meta = carregar_metadados()
    n_teste = len(y_teste)
    n_total = meta.get("n_amostras", len(y))
    subtitulo_amostra = f"Amostra de teste: {n_teste} pacientes (20% de {n_total})"
    cm = confusion_matrix(y_teste, y_pred, labels=CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acuracia = float((y_pred == y_teste).mean())

    _plotar_matriz_confusao(
        cm, cm_norm, n_teste, meta.get("n_amostras", len(y)), acuracia, ARTIFACTS / "confusion_matrix.png"
    )

    taxas = []
    for i, c in enumerate(CLASSES):
        total = cm[i].sum()
        acertos = cm[i, i]
        erros = total - acertos
        taxas.append(
            {
                "classe": NOMES[c],
                "classe_rotulo": f"{NOMES[c]}\n({acertos} acertos · {erros} erros)",
                "taxa_erro": erros / total if total else 0,
                "taxa_acerto": acertos / total if total else 0,
                "acertos": int(acertos),
                "erros": int(erros),
                "total": int(total),
            }
        )
    df_taxas = pd.DataFrame(taxas)
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(df_taxas))
    barras_acerto = ax.bar(x_pos, df_taxas["taxa_acerto"], label="Acertou", color="#2ecc71", width=0.55)
    ax.bar(
        x_pos, df_taxas["taxa_erro"], bottom=df_taxas["taxa_acerto"], label="Errou", color="#e74c3c", width=0.55
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_taxas["classe_rotulo"], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Percentual (0% a 100%)")
    ax.set_title(
        "Resultado por prioridade — amostra de teste\n"
        "Verde = IA igual às regras SIGPS · Vermelho = divergência",
        fontsize=12,
    )
    for bar, row in zip(barras_acerto, df_taxas.itertuples(), strict=True):
        if row.taxa_acerto > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                row.taxa_acerto / 2,
                f"{row.acertos}/{row.total}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
    ax.legend(loc="upper right")
    _figfootnote(fig)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(ARTIFACTS / "taxas_erro_classe.png", dpi=150, bbox_inches="tight")
    plt.close()

    proba_max = y_proba.max(axis=1)
    df_proba = pd.DataFrame(y_proba, columns=[f"IA acha: {NOMES[c]}" for c in modelo.classes_])
    df_proba["classe_real"] = [NOMES_COMPLETOS[c] for c in y_teste]
    df_proba["proba_maxima"] = proba_max

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(proba_max, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Probabilidade máxima")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title(f"Quão segura a IA estava da resposta\n{subtitulo_amostra}")
    sns.violinplot(
        data=df_proba, x="classe_real", y="proba_maxima", ax=axes[1], order=[NOMES_COMPLETOS[c] for c in CLASSES]
    )
    axes[1].set_title("Segurança da IA por prioridade real")
    axes[1].set_xlabel("Prioridade correta (regras SIGPS)")
    axes[1].set_ylabel("Confiança da IA (0 a 1)")
    _figfootnote(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(ARTIFACTS / "distribuicao_probabilidades.png", dpi=150, bbox_inches="tight")
    plt.close()

    proba_media = df_proba.groupby("classe_real").mean(numeric_only=True)
    cols_proba = [c for c in proba_media.columns if c.startswith("IA acha:")]
    proba_exibir = proba_media[cols_proba].rename(
        columns={
            "IA acha: Normal": "Predição: Normal",
            "IA acha: Alta": "Predição: Alta",
            "IA acha: Extrema": "Predição: Extrema",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(proba_exibir, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title(
        "Confiança média da IA — por prioridade real\n"
        "Ex.: pacientes realmente Extrema devem ter alta barra em «Predição: Extrema»",
        fontsize=11,
    )
    ax.set_ylabel("Prioridade correta (regras SIGPS)")
    ax.set_xlabel("O que a IA previu em média")
    _figfootnote(fig)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(ARTIFACTS / "heatmap_probabilidades.png", dpi=150, bbox_inches="tight")
    plt.close()

    idx_extrema = list(modelo.classes_).index(3)
    p_extrema = y_proba[:, idx_extrema]
    y_real_extrema = (y_teste == 3).astype(int)
    limiares = np.linspace(0.05, 0.95, 40)
    recalls, precisions, f1s = [], [], []
    for t in limiares:
        y_bin = (p_extrema >= t).astype(int)
        recalls.append(recall_score(y_real_extrema, y_bin, zero_division=0))
        precisions.append(precision_score(y_real_extrema, y_bin, zero_division=0))
        f1s.append(f1_score(y_real_extrema, y_bin, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(limiares, recalls, label="Encontrou os Extrema reais (Recall)", linewidth=2.5, color="#3498db")
    ax.plot(limiares, precisions, label="Acertou quando marcou Extrema (Precisão)", linewidth=2.5, color="#e67e22")
    ax.plot(
        limiares, f1s, label="Equilíbrio entre os dois (F1)", linewidth=2, linestyle="--", color="#27ae60"
    )
    ax.axvline(1 / 3, color="gray", linestyle=":", linewidth=1.5, label="Regra atual da IA (~33%)")
    ax.set_xlabel(
        "Quanto a IA precisa «ter certeza» para marcar Extrema\n"
        "(0 = aceita qualquer dúvida · 1 = só se tiver 100% de certeza)"
    )
    ax.set_ylabel("Desempenho (0% a 100%)")
    ax.set_title(
        "Sensibilidade da classificação Extrema\n"
        "Se subirmos o exigência, erramos menos falsos alarmes, mas podemos perder urgentes",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    _figfootnote(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(ARTIFACTS / "curva_limiar_extrema.png", dpi=150, bbox_inches="tight")
    plt.close()

    imp = pd.Series(modelo.feature_importances_, index=ORDEM_FEATURES).sort_values(ascending=True)
    rotulos = {
        "idade": "Idade",
        "tem_diabetes": "Diabetes",
        "tem_hipertensao": "Hipertensão",
        "tem_cancer": "Câncer",
        "organization_id": "Clínica (ID)",
    }
    imp.index = [rotulos.get(i, i) for i in imp.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot(kind="barh", color="teal", ax=ax)
    ax.set_title("O que mais influencia a prioridade na IA")
    ax.set_xlabel("Importância relativa (quanto maior, mais a IA usa na decisão)")
    _figfootnote(fig, "Variáveis enviadas pela ficha clínica do SIGPS à API de priorização.")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(ARTIFACTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    validacao_sigps = _gerar_validacao_casos_sigps(modelo)

    # Remove gráficos da versão posterior (5 figuras extras)
    for extra in (
        "distribuicao_classes_teste.png",
        "metricas_por_classe.png",
        "validacao_cruzada.png",
        "comparacao_modelos.png",
        "efeito_idade_prioridade.png",
    ):
        path_extra = ARTIFACTS / extra
        if path_extra.exists():
            path_extra.unlink()

    contexto = {
        "tipo_avaliacao": "fidelidade_ao_protocolo_sigps",
        "dataset": meta.get("nome", "SIGPS Demo Cohort 1000"),
        "fonte": meta.get("fonte", "Banco SIGPS — Clínica Demo Fila IA"),
        "descricao": (
            "Mede se o Random Forest reproduz o protocolo de triagem SIGPS "
            f"(idade + diabetes + hipertensão + câncer) na amostra de teste ({n_teste} pacientes). "
            "Não substitui validação clínica independente."
        ),
        "nota_100_porcento": NOTA_AVALIACAO,
        "n_teste": int(n_teste),
        "n_total_cohort": meta.get("n_amostras"),
        "acuracia_vs_rotulo_protocolo": float((y_pred == y_teste).mean()),
        "fidelidade_ml_vs_regras_sigps": fidelidade_regras,
        "validacao_casos_demo_sigps": validacao_sigps,
    }
    (ARTIFACTS / "metrics_sigps_context.json").write_text(
        json.dumps(contexto, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gerados = sorted(p.name for p in ARTIFACTS.glob("*.png"))
    print("Gráficos gerados em artifacts/:")
    for nome in gerados:
        print(f"  - {nome}")
    print(f"Total: {len(gerados)} arquivos PNG")
    print(f"Contexto SIGPS: {ARTIFACTS / 'metrics_sigps_context.json'}")
    print(f"Casos demo SIGPS: {validacao_sigps['acertos_ml']}/{validacao_sigps['n_casos']} acertos ML")


if __name__ == "__main__":
    main()
