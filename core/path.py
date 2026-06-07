from pathlib import Path

PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_ARTIFACTS = PASTA_RAIZ / "artifacts"
PASTA_DATA_PROCESSED = PASTA_RAIZ / "data" / "processed"
CAMINHO_COHORT_CSV = PASTA_DATA_PROCESSED / "sigps_demo_cohort_1000.csv"
CAMINHO_COHORT_META = PASTA_DATA_PROCESSED / "sigps_demo_cohort_meta.json"
CAMINHO_MODELO = PASTA_ARTIFACTS / "model.pkl"
CAMINHO_AVALIACAO = PASTA_ARTIFACTS / "metrics_eval.json"
CAMINHO_METRICAS = PASTA_ARTIFACTS / "metrics.json"