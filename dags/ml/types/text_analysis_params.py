from typing import List, Optional, Union

from pydantic import BaseModel, validator


class TextWord2VecParams(BaseModel):
    fillna: Optional[str]
    dropna: bool
    dropNonLitChars: bool
    stopWords: Union[str, List[str]]
    regexTokenizer: str
    vectorSize: int
    windowSize: int
    minCount: int
    numPartitions: int
    maxSentenceLength: int
    stepSize: float
    maxIter: int
    seed: int
    inputCol: str
    outputCol: str
    returnTempColumns: bool


class TextClusteringParams(BaseModel):
    min_cluster_num: int
    max_cluster_num: int
    feature_cols: Union[str, List[str]]
    prediction_col: str
    evaluation: str
    clustering_seed: int
    cluster_frequency: str

    @validator("cluster_frequency")
    def check_cluster_frequency(cls, v):
        assert v in ("unique_processes", "coefficient"), (
            f"cluster frequency must be in ('unique_processes', 'coefficient'),"
            f" but got {v}"
        )


class TextAnalysisParams(BaseModel):
    text_col: str
    user_clusters: Optional[int]
    remove_duplicates: bool
    clusters_descriptions: List[str]
    cluster_words_amount: int
    other_in_general_cluster: bool
    num_features: int
    n_partitions: int
    word2vec_params: TextWord2VecParams
    clustering_params: TextClusteringParams

    @validator("user_clusters")
    def user_clusters_from_2_to_10(cls, v):
        if v is None:
            return v
        assert 2 <= v <= 10, (
            "user_clusters must be from 2 to 10"
        )
        return v
