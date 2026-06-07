"""Protocolo de triagem SIGPS — usado pela API quando o model.pkl não está disponível."""


def calcular_prioridade(idade: int, tem_diabetes: int, tem_hipertensao: int, tem_cancer: int) -> int:
    """Níveis: 1 Normal, 2 Alta, 3 Extrema."""
    if tem_cancer == 1 or (idade >= 60 and (tem_diabetes == 1 or tem_hipertensao == 1)) or (
        idade > 50 and tem_diabetes == 1 and tem_hipertensao == 1
    ):
        return 3
    if idade >= 60 or tem_diabetes == 1 or tem_hipertensao == 1:
        return 2
    return 1
