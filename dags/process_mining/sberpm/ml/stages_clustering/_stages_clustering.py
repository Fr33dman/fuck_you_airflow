from re import sub
from typing import Union

from numpy import full, inf
from pandas import DataFrame, Series, concat, merge

from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from navec import Navec

from tqdm.notebook import tqdm

from process_mining.sberpm import models
from process_mining.sberpm._holder import DataHolder


class StagesClustering:
    """
    The clustering module of the process stages.

    Parameters:

        data: DataHolder or DataFrame
            Object that contains the event log and the names of its necessary columns.

        stages_col: str
            A data column containing the text names of the process stages.

        generalizing_ability: float value within [0;1]
            The coefficient of generalizing ability. The higher the coefficient,
            the larger the DBSCAN radius will be, the more elements there will be in the cluster.

        type_model_w2v: bool
            Model Type True = "navec" or False = "PM" (Process Mining).

    Methods:
        apply() - start the clustering process.

        get_clustered_result() - returns a DataFrame with 2 columns:
                                    * {stages_col} - process stage names,
                                    * `Номер кластера (-1 — выбросы)` - cluster name

        get_result_data() - returns the original Data Frame with the addition of the "Номер кластера (-1 — выбросы)" column.

    """

    def __init__(
        self,
        data: Union[DataFrame, DataHolder],  # DataFrame or DataHolder type
        stages_col: str,
        generalizing_ability: float = 0.5,
        type_model_w2v: bool = True,
    ):

        if isinstance(data, DataFrame):
            self._data = data.copy()
        elif isinstance(data, DataHolder):
            self._data = data.data.copy()
        else:
            raise ValueError("Unknown data format. The data format must be of the DataFrame or DataHolder type")

        if not (0 <= generalizing_ability <= 1):
            raise ValueError("'generalizing_ability' must belong to the range from 0 to 1")

        if stages_col not in self._data.columns:
            raise KeyError("Unknown column: ", stages_col)

        self._type_model_w2v = type_model_w2v
        self._stages_col = stages_col
        self._dbscan_eps = 0.1 + 0.65 * generalizing_ability
        self._result = DataFrame()

        # это функция векторизации с помощью Word2Vec
        if self._type_model_w2v:
            self._model_w2vl = Navec.load(models.get_navec_model()).as_gensim

        else:
            self._model_w2vl = Word2Vec.load(models.get_PM_model())

    @staticmethod
    def lemmatize_text(text, segmenter, morph_vocab, morph_tagger):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        return " ".join([lemma.lemma for lemma in doc.tokens])

    def apply(self):
        """
        Start the clustering process.
        """

        self._result: DataFrame = self._data[self._stages_col].drop_duplicates()
        self._result.dropna(inplace=True)
        self._result = self._result[~self._result.isin([" ", ".", "..."])].reset_index(drop=True)

        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)

        for message in self._result.unique():
            cleared_message = sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", " ", message)  # from string.punctuation
            cleared_message = sub(r"\d", " ", cleared_message)  # from string.punctuation

        distances = full(shape=(self._result.shape[0], self._result.shape[0]), fill_value=100000)

        for i in tqdm(range(self._result.shape[0])):
            for j in range(self._result.shape[0]):
                distance = self._model_w2vl.wmdistance(self._result.iloc[i], self._result.iloc[j])
                distances[i, j] = distance if distance != inf else 100000

        # We need to specify algoritum='brute' as the default assumes a continuous feature space.
        dbscan = DBSCAN(eps=self._dbscan_eps, metric="precomputed", algorithm="brute", min_samples=2)
        clusters = Series(data=dbscan.fit_predict(distances), name="Номер кластера (-1 — выбросы)")
        self._result = concat([self._result, clusters], axis=1)

    def get_clustered_result(self) -> DataFrame:
        """
        Returns
        -------
        Returns a pandas DataFrame specifying the cluster of each stage
        """
        if not self._result.empty:
            return self._result.copy()
        else:
            raise RuntimeError("Call apply() method for StagesClustering object first.")

    def get_result_data(self) -> DataFrame:
        """
        Returns
        -------
        Returns the original pandas DataFrame with the added `Номер кластера (-1 — выбросы)` column with cluster names
        """
        if not self._result.empty:
            return merge(left=self._data, right=self._result, on=self._stages_col, how="left")
        else:
            raise RuntimeError("Call apply() method for StagesClustering object first.")
