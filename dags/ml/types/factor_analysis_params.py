from typing import List

from pydantic import BaseModel, validator


class FactorAnalysisParams(BaseModel):
    input_features: List[str]
    target_col: str
    n_partitions: int
    interpretation: str
    algorithm: str

    @validator("interpretation")
    def interpretation_feat_or_import(cls, v):
        assert v in ("FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"), (
            f"interpretation must be 'FEATURE_IMPORTANCE' or "
            f"'PERMUTATION_IMPORTANCE', but got {v}"
        )
        return v

    @validator("algorithm")
    def algorithm_xgboost_or_rf(cls, v):
        assert v in ("XGBOOST", "RANDOM_FOREST"), (
            f"algorithm must be 'XGBOOST' or "
            f"'RANDOM_FOREST', but got {v}"
        )
        return v
