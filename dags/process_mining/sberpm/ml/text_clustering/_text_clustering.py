import os
import re
import string
from functools import lru_cache
from itertools import combinations
from typing import Optional, List, Union

import Levenshtein
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from navec import Navec
from nltk.stem.snowball import RussianStemmer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from process_mining.sberpm import models as _models
from process_mining.sberpm._holder import DataHolder


class TextClustering:
    def __init__(
        self,
        data: Union[pd.DataFrame, DataHolder],  # DataFrame or DataHolder type
        description: str,
        pca_dim: str = "medium",  # fast, medium, full_quality
        dbscan_eps: float = 0.5,
        type_model_w2v: bool = True,
        min_samples: int = 2,
        only_unique_descriptions: bool = False,  # [True, False] рассматриваются только уникальные описания
        cluster_marking: Optional[List[str]] = None,
    ):

        if isinstance(data, pd.DataFrame):
            self._data = data.copy()
        elif isinstance(data, DataHolder):
            self._data = data.data.copy()
        else:
            raise ValueError("Unknown data format. The data format must be of the DataFrame or DataHolder type")

        if pca_dim == "fast":
            self._pca_dim = 3
        elif pca_dim == "medium":
            self._pca_dim = 7
        elif pca_dim == "full_quality":
            self._pca_dim = 13
        else:
            raise ValueError(
                "An incorrect value was entered for pca_dim. pca_dimm takes the values 'low', 'medium' "
                "or 'high'."
            )

        self._clustered_data = pd.DataFrame()
        self._description = description
        self._w2v_representation_columns = list(range(self._pca_dim))
        self._dbscan_eps = dbscan_eps
        self._type_model_w2v = type_model_w2v

        if min_samples < 2:
            raise ValueError(f"'min_samples' got the value {min_samples}. The minimum value for min_samples is 2.")

        self._min_samples = min_samples
        self._only_unique_descriptions = only_unique_descriptions
        self._cluster_marking = cluster_marking

        self._outliers_value = 100000

        # read stopwords
        with open(os.path.join(os.path.dirname(__file__), "stopwords_ru.txt"), encoding="utf-8") as file:
            self._stopwords_ru = file.read().split("\n")
        with open(os.path.join(os.path.dirname(__file__), "stopwords_en.txt")) as file:
            self._stopwords_en = file.read().split("\n")

        self._stopwords = self._stopwords_ru + self._stopwords_en  # from nltk stopwords

        # это функция векторизации с помощью Word2Vec
        if self._type_model_w2v:
            self._model_w2vl = Navec.load(_models.get_navec_model()).as_gensim
            self._w2v_size = 300
        else:
            self._model_w2vl = Word2Vec.load(_models.get_PM_model())
            self._w2v_size = 64

        self._result = pd.DataFrame()

    def _clear_data(self):
        # оставляем descriptions только по нужным колонкам
        self._clustered_data = self._clustered_data[
            (self._clustered_data[self._description].notna())
            & (
                ~self._clustered_data[self._description].isin(
                    [" ", ".", "!", ",", "*", "-", "+", "..", "...", "/"]
                )
            )
        ]

        if self._only_unique_descriptions:
            self._clustered_data = self._clustered_data.drop_duplicates(subset=[self._description])
        self._clustered_data = self._clustered_data.reset_index()  # для нормальной работы конкатенации

    def _vectorize_w2v(self, message):
        words = np.nanmean(
            [
                lru_cache(maxsize=50000)(lambda word: self._model_w2vl[word])(word)
                for word in message
                if word in self._model_w2vl.wv.vocab
            ],
            axis=0,
        )
        if words.size == 1:
            words = np.zeros((self._w2v_size,))
        return words

    # Удаление окончаний, знаков и цифр, разделение на слова, приведение к нижнему регистру.
    # Рузультаты стемминга кэшируются, что дает ускорение в 100 раз
    @staticmethod
    def _normalize_pm(message):
        message = re.sub(r"[^(\w\d)]+", " ", message)

        words = message.split(" ")

        stemmer = RussianStemmer()
        cached_stemmer = lru_cache(maxsize=50000)(stemmer.stem)
        words = [cached_stemmer(word) for word in words]

        words = [word.lower() for word in words if word.isalpha()]

        return words

    # Удаление знаков и цифр, разделение на слова, приведение к нижнему регистру.
    @staticmethod
    def _normalize_navec(message):
        message = re.sub(r"[^(\w\d)]+", " ", message)

        words = message.split(" ")

        words = [word.lower() for word in words if word.isalpha()]

        return words

    # нормализация всех наблюдений в self._clustered_data, W2V векторизация, понижение размерности с помощью PCA
    def _prepare_messages_PCA(self, data, text_column):

        if self._type_model_w2v:
            data["normalized_messages"] = data[text_column].map(self._normalize_navec)
        else:
            data["normalized_messages"] = data[text_column].map(self._normalize_pm)

        messages_enc = [self._vectorize_w2v(message) for message in data["normalized_messages"]]

        pca = PCA(n_components=self._pca_dim, random_state=42)

        low_dimmed = pca.fit_transform(np.array(messages_enc))

        return low_dimmed

    def _cluster_descriptions(self, n_dim):
        """
        DBSCAN version of clusterization
        """
        try:
            to_clust = self._clustered_data[list(range(n_dim))]

            std_slc = StandardScaler()
            to_clust_std = std_slc.fit_transform(to_clust)

            dbscan = DBSCAN(eps=self._dbscan_eps, min_samples=self._min_samples)
            model = dbscan.fit(to_clust_std)

            self._clustered_data["cluster"] = model.fit_predict(to_clust_std)
        except KeyError:
            print("KeyError")

    @staticmethod
    def _normalize_2(message):
        message = re.sub(r"[\W]", " ", message)
        message = re.sub("_+", " ", message)
        words = message.split(" ")
        words = [word.lower() for word in words]
        return words

    @staticmethod
    def _normalize_3(message):
        message = re.sub(r"[^(\w\d)]+", " ", message)
        words = message.split(" ")
        words = [word.lower() for word in words]
        return words

    @staticmethod
    def _cut_digits_symbols(message):
        message = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "", message)  # email
        message = re.sub(r"[^(\w | \. | , | : | - )]", " ", message)
        message = re.sub(":", " ", message)
        message = re.sub("\d+", " ", message)
        message = re.sub(" \.", "", message)
        message = re.sub("\.+", ". ", message)
        message = re.sub(" ,", ", ", message)
        message = re.sub(",+", ", ", message)
        message = re.sub(" +", " ", message)
        message = re.sub(r"[\s,\.][a-zA-Z][\s,\.]", " ", message)
        return message

    @staticmethod
    def _cut_personal_data(message, word):
        message = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "", message)  # email
        message = re.sub(r"\s+", " ", message)
        message = re.sub(" " + word.capitalize() + "[^(\w)]", " ", message)
        message = re.sub("^" + word.capitalize() + "[^(\w)]", "", message)
        message = re.sub("[^(\w)]" + word.capitalize() + "$", "", message)
        first_character_lower = lambda s: s[:1].lower() + s[1:] if s else ""
        message = re.sub(" " + first_character_lower(word) + "[^(\w)]", " ", message)
        message = re.sub(" " + word.upper() + " ", " ", message)
        message = re.sub(" " + word.upper() + ",", ",", message)
        message = re.sub(" " + word.upper() + "\.", "", message)
        message = re.sub("^" + first_character_lower(word) + "[^(\w)]", "", message)
        message = re.sub("^" + word.upper() + "[^(\w)]", "", message)
        message = re.sub("[^(\w)]" + first_character_lower(word) + "$", "", message)
        message = re.sub("[^(\w)]" + word.upper() + "$", "", message)
        message = re.sub(r"[ ,\.](none|NONE|None|NULL|null|Null)[ ,\.]", " ", message)
        message = re.sub(r",\s*\)", "\)", message)
        message = re.sub(r"\.+,+ *", "\. ", message)
        message = re.sub(r" *\)", "\)", message)

        return message

    @staticmethod
    def _final_cleaning(message):
        message = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "", message)  # email
        message = re.sub(r"\s+", " ", message)
        message = re.sub(r", +\.+ +,+ +,+ +\.+", ". ", message)
        message = re.sub(r", $", "", message)
        message = re.sub(r"_+", " ", message)
        message = re.sub(r"\s\W*$", "", message)
        message = re.sub(r"^\W*", "", message)
        return message

    # функция для нахождения самых популярных слов для кластера

    def _most_popular_words(self, data: pd.DataFrame, text_column: str, n: int):
        temp_data = pd.DataFrame()
        temp_data["normalized_messages2"] = data[text_column].apply(lambda message: self._normalize_2(message))
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        vectorizer.fit_transform(temp_data["normalized_messages2"].astype(str))
        return list(vectorizer.vocabulary_.keys())[:n]

    def _count_words(self, data, text_column):
        temp_data = pd.DataFrame()
        temp_data["normalized_messages3"] = data[text_column].apply(lambda message: self._normalize_3(message))
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        fitted = vectorizer.fit_transform(temp_data["normalized_messages3"].astype(str))
        count_list = np.asarray(fitted.sum(axis=0))[0]

        return sum(count_list)

    # функция возвращает словарь {одно слово из корпуса data(обычно подмножество столбца) :
    # количество слова в корпус data(обычно подмножество столбца)}
    def _frequency_of_words(self, data, text_column):
        data["normalized_messages2"] = [self._normalize_2(message) for message in data[text_column]]
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        fitted = vectorizer.fit_transform(data["normalized_messages2"].astype(str))
        word_list = vectorizer.get_feature_names()
        count_list = np.asarray(fitted.sum(axis=0))[0]
        return dict(sorted(dict(zip(word_list, count_list)).items(), key=lambda item: item[1]))

    @staticmethod
    def _messages_identity(data, text_column):
        for idx in range(len(data[text_column]) - 1):
            if Levenshtein.distance(data[text_column].values[idx], data[text_column].values[idx + 1]) > 2:
                return False
        return True

    def _consists_of_stopwords(self, message):
        words = re.sub("[" + string.punctuation + "]", "", message).split()
        for i in words:
            if i.lower() not in self._stopwords:
                return False
        return True

    def apply(self):

        self._clustered_data = self._data.copy()
        self._clear_data()
        marked_df = pd.DataFrame()

        self._clustered_data["closest_to_cluster"] = np.NaN
        self._clustered_data["cluster"] = np.NaN
        self._clustered_data["cluster"] = self._clustered_data["cluster"].astype("Int64")
        self._clustered_data["closest_to_cluster"] = self._clustered_data["closest_to_cluster"].astype("Int64")
        # функция кластеризации для определенных function_code, возвращает self._clustered_data уже с кластерами

        if self._cluster_marking:
            distances = np.full(
                shape=(self._clustered_data.shape[0], len(self._cluster_marking)), fill_value=self._outliers_value
            )

            for i in range(self._clustered_data.shape[0]):
                for j in range(len(self._cluster_marking)):
                    distance = self._model_w2vl.wmdistance(
                        self._clustered_data[self._description].iloc[i], self._cluster_marking[j]
                    )
                    distances[i, j] = distance if distance != np.inf else self._outliers_value

            self._clustered_data = pd.concat(
                [self._clustered_data, pd.DataFrame(data=distances, columns=self._cluster_marking)], axis=1
            )

            dbscan = DBSCAN(
                eps=0.33, algorithm="brute", min_samples=(2 if self._only_unique_descriptions == True else 5)
            )
            model = dbscan.fit(distances)
            self._clustered_data["marked_cluster"] = model.fit_predict(distances)

            closest_clustID_to_marks = (
                self._clustered_data[self._cluster_marking + ["marked_cluster"]]
                .groupby(by="marked_cluster")
                .mean()
                .idxmin(axis=0)
            )

            marked_df = self._clustered_data[
                self._clustered_data["marked_cluster"].isin(closest_clustID_to_marks.values)
            ]
            self._clustered_data = self._clustered_data[
                ~self._clustered_data["marked_cluster"].isin(closest_clustID_to_marks.values)
            ]
            self._clustered_data["marked_cluster"] = np.NaN

            le = LabelEncoder()
            le.fit(closest_clustID_to_marks)
            marked_df["cluster"] = -le.transform(marked_df["marked_cluster"]) - 2

            marked_df.reset_index(drop=True, inplace=True)
            self._clustered_data.reset_index(drop=True, inplace=True)

            marked_df["top_10"] = np.NaN
            marked_df["cluster_meaning"] = np.NaN
            marked_df["closest"] = np.NaN

            for mark in self._cluster_marking:
                clustID = closest_clustID_to_marks[mark]

                top_10 = self._most_popular_words(
                    marked_df[marked_df["marked_cluster"] == clustID], self._description, 10
                )

                marked_df["top_10"][marked_df["marked_cluster"] == clustID] = str(top_10)

                marked_df["cluster_meaning"][marked_df["marked_cluster"] == clustID] = mark

                closest_descripton_index = marked_df[mark][marked_df["marked_cluster"] == clustID].idxmin(axis=0)
                marked_df["closest"][marked_df["marked_cluster"] == clustID] = marked_df[self._description].iloc[
                    closest_descripton_index
                ]

        low_dimmed = self._prepare_messages_PCA(self._clustered_data, self._description)
        self._clustered_data = self._clustered_data.reset_index()  # для нормальной работы конкатенации
        self._clustered_data = pd.concat([self._clustered_data, pd.DataFrame(data=low_dimmed)], axis=1)

        self._cluster_descriptions(self._pca_dim)

        for clustID in self._clustered_data["cluster"].unique():
            centroid = self._clustered_data.loc[:, self._w2v_representation_columns][
                self._clustered_data["cluster"] == clustID
            ].mean(axis=0)

            closest_pos_by_clust, distance = pairwise_distances_argmin_min(
                [centroid],
                self._clustered_data.loc[:, self._w2v_representation_columns][
                    self._clustered_data["cluster"] == clustID
                ],
            )

            closest_pos_by_df = self._clustered_data.loc[:, self._w2v_representation_columns][
                self._clustered_data["cluster"] == clustID
            ].index[closest_pos_by_clust]

            self._clustered_data["closest_to_cluster"].iloc[closest_pos_by_df] = clustID

        self._clustered_data["top_10"] = np.NaN
        self._clustered_data["cluster_meaning"] = np.NaN
        self._clustered_data["closest"] = np.NaN

        for clustID in self._clustered_data["cluster"].unique():

            top_10 = self._most_popular_words(
                self._clustered_data[self._clustered_data["cluster"] == clustID], self._description, 10
            )

            self._clustered_data["top_10"][self._clustered_data["cluster"] == clustID] = str(top_10)

            _frequency_of_words_in_cluster = self._frequency_of_words(
                self._clustered_data[self._clustered_data["cluster"] == clustID], self._description
            )
            cluster_meaning = self._clustered_data[self._description][
                self._clustered_data["closest_to_cluster"] == clustID
            ].item()

            self._clustered_data["closest"][self._clustered_data["cluster"] == clustID] = str(cluster_meaning)
            n_words = self._count_words(
                self._clustered_data[self._clustered_data["cluster"] == clustID], self._description
            )

            to_kmeans = list(_frequency_of_words_in_cluster.values()) + [
                1 for _ in range(int((n_words - sum(list(_frequency_of_words_in_cluster.values()))) / 2))
            ]

            # Кластеризуем слова на частые и нечастые по количеству вхождений в self._description для каждого
            # cluster_unique

            if (
                len(to_kmeans) == 1
                or self._messages_identity(
                    self._clustered_data[self._clustered_data["cluster"] == clustID], self._description
                )
                == True
            ):

                cluster_meaning = self._final_cleaning(self._cut_digits_symbols(cluster_meaning))
                self._clustered_data["cluster_meaning"][self._clustered_data["cluster"] == clustID] = str(
                    cluster_meaning
                )
            else:
                kmeans = KMeans(n_clusters=2, random_state=42).fit(np.expand_dims(to_kmeans, axis=-1))

                for idx in range(len(list(_frequency_of_words_in_cluster.values()))):
                    label = kmeans.labels_[idx]

                    cluster_meaning = self._cut_digits_symbols(cluster_meaning)

                    if label == int(kmeans.cluster_centers_[0] >= kmeans.cluster_centers_[1]):
                        cluster_meaning = self._cut_personal_data(
                            cluster_meaning, list(_frequency_of_words_in_cluster.keys())[idx]
                        )
                cluster_meaning = self._final_cleaning(cluster_meaning)
                self._clustered_data["cluster_meaning"][self._clustered_data["cluster"] == clustID] = str(
                    cluster_meaning
                )

        self._clustered_data["merged_cluster"] = np.NaN
        self._clustered_data["merged_cluster"] = self._clustered_data["merged_cluster"].astype("Int64")

        # Ищем нулевые или состоящие из стоп-слов cluster_meaning и записываем их в -1 кластер выбросов
        for cluster in self._clustered_data["cluster"].unique()[self._clustered_data["cluster"].unique() != -1]:
            cluster_meaning = self._clustered_data["cluster_meaning"][
                self._clustered_data["cluster"] == cluster
            ].values[0]
            if (cluster_meaning == "") or self._consists_of_stopwords(cluster_meaning):
                self._clustered_data["merged_cluster"][self._clustered_data["cluster"] == cluster] = -1
                self._clustered_data["closest_to_cluster"][self._clustered_data["cluster"] == cluster] = np.NaN

        # схлопываем кластеры, у которых один и тот же смысл
        for cluster1, cluster2 in combinations(self._clustered_data["cluster"].unique(), 2):
            cluster_meaning1 = self._clustered_data["cluster_meaning"][
                self._clustered_data["cluster"] == cluster1
            ].values[0]
            cluster_meaning2 = self._clustered_data["cluster_meaning"][
                self._clustered_data["cluster"] == cluster2
            ].values[0]
            if cluster_meaning1 != "" and cluster_meaning2 != "":
                if (
                    Levenshtein.distance(cluster_meaning1.lower(), cluster_meaning2.lower())
                    / max(len(cluster_meaning1), len(cluster_meaning2))
                    < 0.3
                ):

                    if (
                        self._clustered_data["merged_cluster"][self._clustered_data["cluster"] == cluster1]
                        .notna()
                        .all()
                    ):
                        self._clustered_data["merged_cluster"][
                            self._clustered_data["cluster"] == cluster2
                        ] = self._clustered_data["merged_cluster"][
                            self._clustered_data["cluster"] == cluster1
                        ].iloc[
                            0
                        ]
                    else:
                        self._clustered_data["merged_cluster"][
                            self._clustered_data["cluster"] == cluster2
                        ] = cluster1

                    self._clustered_data["closest_to_cluster"][
                        self._clustered_data["closest_to_cluster"] == cluster2
                    ] = np.NaN
                    self._clustered_data["cluster_meaning"][
                        self._clustered_data["cluster"] == cluster2
                    ] = cluster_meaning1

        # Обновляем исходный номер кластера для тех кластеров, которые выбраны для слияния
        for i in range(len(self._clustered_data["merged_cluster"])):
            if pd.notna(self._clustered_data["merged_cluster"].iloc[i]):
                self._clustered_data["cluster"].iloc[i] = self._clustered_data["merged_cluster"].iloc[i]

        # Обновляем столбец с ближайшими сообщениями к центроиду кластера
        for clustID in self._clustered_data["cluster"].unique():
            cluster_meaning = self._clustered_data[self._description][
                self._clustered_data["closest_to_cluster"] == clustID
            ].item()

            self._clustered_data["closest"][self._clustered_data["cluster"] == clustID] = str(cluster_meaning)

        # выравнивание номеров кластеров после схлопывания кластеров
        le = LabelEncoder()
        if -1 in self._clustered_data["cluster"].unique():
            self._clustered_data["cluster"] = le.fit_transform(self._clustered_data["cluster"]) - 1
            self._clustered_data["cluster_meaning"][self._clustered_data["cluster"] == -1] = "Выбросы"
        else:
            self._clustered_data["cluster"] = le.fit_transform(self._clustered_data["cluster"])

        # Посчитаем заново 10 самых популярных слов по кластерам, так как состав этих кластеров изменился
        for clustID in self._clustered_data["cluster"].unique():
            top_10 = self._most_popular_words(
                self._clustered_data[self._clustered_data["cluster"] == clustID], self._description, 10
            )

            self._clustered_data["top_10"][self._clustered_data["cluster"] == clustID] = str(top_10)
        if self._cluster_marking:
            output_df = self._clustered_data.drop(
                columns=[
                    "normalized_messages",
                    "closest_to_cluster",
                    "marked_cluster",
                    "merged_cluster",
                    "level_0",
                    "index",
                ]
                + self._w2v_representation_columns
                + self._cluster_marking,
                errors="ignore",
            )

            output_marked_df = marked_df.drop(
                columns=["marked_cluster", "closest_to_cluster", "level_0", "index"] + self._cluster_marking,
                errors="ignore",
            )
            output = pd.concat([output_marked_df, output_df], axis=0)

            # пишем результаты кластеризации в файл
            output.rename(
                columns={
                    "cluster": "Номер кластера (положительные — автоматические, отрицательные — ручные, -1 — выбросы)",
                    "top_10": "10 самых популярных слов в кластере",
                    "closest": "Ближайшее сообщение к смыслу кластера",
                    "cluster_meaning": "Смысл кластера",
                }
            )
        else:
            output = self._clustered_data

        # запишем в файл только информацию про текстовый анализ
        self._result = output[["cluster", "cluster_meaning", self._description, "closest", "top_10"]].sort_values(
            by=["cluster", self._description]
        )

        self._result = self._result.rename(
            columns={
                "cluster": "Номер кластера (положительные — автоматические, отрицательные — ручные, -1 — выбросы)",
                "top_10": "10 самых популярных слов в кластере",
                "closest": "Ближайшее сообщение к смыслу кластера",
                "cluster_meaning": "Смысл кластера",
            }
        )

    def get_result(self):
        if not self._result.empty:
            return self._result.copy()
        else:
            raise RuntimeError("Call apply() method for autoInsights object first.")
