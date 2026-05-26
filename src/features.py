from dataclasses import dataclass
import numpy as np

ORDEM_FEATURES = [
    "idade",
    "tem_diabetes",
    "tem_hipertensao",
    "tem_cancer",
]

@dataclass(frozen=True)
class FeaturesPrioridade:
    idade: int
    tem_diabetes: int
    tem_hipertensao: int
    tem_cancer: int

    def para_array(self) -> np.ndarray:
        return np.array(
            [
                self.idade,
                self.tem_diabetes,
                self.tem_hipertensao,
                self.tem_cancer,
            ],
            dtype=np.float32,
        )
    
def para_entrada_modelo(features: FeaturesPrioridade) -> np.ndarray:
    return features.para_array().reshape(1, -1)