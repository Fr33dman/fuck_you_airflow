from typing import Dict, List

from pydantic import ValidationError

from ml.types import DatasetParams, NotationParams, \
    AutoInsightsParams, FactorAnalysisParams, TextAnalysisParams, \
    TextWord2VecParams, TextClusteringParams, ResearchParams
from ml.universal_model_interface.data_types.types.python_model_type import PythonModelParams
from ml.universal_model_interface.data_types.types.front_response_type import FrontPythonModelResponse


def setup_model_params(data: FrontPythonModelResponse, meta_info: Dict, stripped: bool = True) -> PythonModelParams:
    model_params = None

    for model_info in meta_info.get("models"):
        if model_info.get("modelName") == data.modelName:
            model_info["modelLabel"] = data.modelLabel
            model_params = PythonModelParams(**model_info)
            break

    for old_param in model_params.modelParams:
        if old_param.paramType == "list":
            value_from_front = data.params.get(old_param.paramName)

            if value_from_front and type(value_from_front) == list:
                striped_data = [a.strip('"') if type(a) == str and stripped else a for a in value_from_front]
                old_param.paramValue = striped_data

            elif type(value_from_front) == str and stripped:
                old_param.paramValue = value_from_front.strip('"')

            elif isinstance(value_from_front, str):
                old_param.paramValue = value_from_front

            else:
                old_param.paramValue = []

        else:
            param = data.params.get(old_param.paramName)

            if isinstance(param, str) and stripped:
                param = param.strip('"')
            old_param.paramValue = param

    return model_params


def setup_research_params(json_java: Dict) -> ResearchParams:
    """
    Setting research params
    Parameters
    ----------
    json_java : Dict
        Json from Java backend
    Returns
    -------

    """
    widget_id = json_java.get("widgetId")
    workspace_id = json_java.get("workspaceId")

    return ResearchParams(
        widget_id=widget_id,
        workspace_id=workspace_id
    )


def setup_dataset_params(json_java: Dict) -> DatasetParams:
    """
    Setting dataset params
    Parameters
    ----------
    json_java : Dict
        Json from Java backend

    Returns
    -------
    DatasetParams
    """
    dataset_path = json_java.get("sourceFile")
    database_name = json_java.get("databaseName")

    dataset_path = database_name + "." + dataset_path

    date_pattern = json_java.get("datePattern")

    return DatasetParams(
        path=dataset_path,
        date_format=date_pattern,
    )


def setup_notation_params(json_java: Dict, stripped: bool = True) -> NotationParams:
    """
    Setting notation params
    Parameters
    ----------
    json_java : Dict
        Json from Java backend
    stripped: bool
        Strip columns or not

    Returns
    -------
    NotationParams
    """
    if stripped:
        temp = {x["keyType"]: x["name"].strip('"')
                for x in json_java.get("notationData")
                if "keyType" in x and "name" in x}
    else:
        temp = {x["keyType"]: x["name"]
                for x in json_java.get("notationData")
                if "keyType" in x and "name" in x}

    id_col = temp.get("ID")
    if id_col is None:
        raise AttributeError("id_col is None")

    date_col = temp.get("DATE", None)
    date_end_col = temp.get("DATE_END", None)
    if date_col is None and date_end_col is None:
        raise ValidationError('Date start column or date end column are required.')

    status_col = temp.get("STATUS_NAME")
    if status_col is None:
        raise AttributeError("status_col is None")

    user_id_col = temp.get("USER_ID")
    if user_id_col is not None:
        user_id_col = user_id_col.rstrip("\r")

    return NotationParams(
        id_col=id_col.rstrip("\r"),
        date_col=date_col if date_col is None else date_col.rstrip("\r"),
        date_end_col=date_end_col if date_end_col is None else date_end_col.rstrip("\r"),
        status_col=status_col.rstrip("\r"),
        user_id_col=user_id_col,
    )


def setup_auto_insights_params(json_java: Dict) -> AutoInsightsParams:
    """
    Setting dataset params
    Parameters
    ----------
    json_java : Dict
        Json from Java backend

    Returns
    -------
    AutoInsightsParams
    """
    insight_type = json_java.get("insightType", "TIME")
    if insight_type not in ("TIME", "CYCLES", "OVERALL"):
        raise ValidationError(
            f"Expected insightType from ('TIME', 'CYCLES', 'OVERALL'), "
            f"but got: {insight_type}"
        )
    qmin = json_java.get("qmin")
    qmax = json_java.get("qmax")
    n_partitions = json_java.get("n_partitions")

    return AutoInsightsParams(
        insight_type=insight_type,
        qmin=float(qmin),
        qmax=float(qmax),
        n_partitions=n_partitions,
    )


def setup_factor_analysis_params(json_java: Dict) -> FactorAnalysisParams:
    """
    Setting dataset params
    Parameters
    ----------
    json_java : Dict
        Json from Java backend

    Returns
    -------
    FactorAnalysisParams
    """
    ml_keys = json_java.get("mlKeys")
    input_features = ml_keys.get("features")
    if input_features:
        input_features = [
            feature["name"].strip('"')
            for feature in input_features
        ]

    target_col = ml_keys.get("target")
    if target_col:
        target_col = target_col["name"].strip('"')

    interpretation = json_java.get("interpretation", "FEATURE_IMPORTANCE")
    algorithm = json_java.get("algorithm", 'XGBOOST')

    n_partitions = json_java.get("n_partitions")

    return FactorAnalysisParams(
        input_features=input_features,
        target_col=target_col,
        n_partitions=n_partitions,
        interpretation=interpretation,
        algorithm=algorithm
    )


def setup_text_analysis_params(json_java: Dict,
                               text_config: Dict) \
        -> TextAnalysisParams:
    """
    Setting dataset params
    Parameters
    ---------
    json_java : Dict
        Json from Java backend
    text_config : Dict
        Params from config

    Returns
    -------
    TextAnalysisParams
    """
    temp = {x["keyType"]: x["name"].strip('"')
            for x in json_java.get("notationData")
            if "keyType" in x and "name" in x}
    text_col = temp.get("OTHER")
    if text_col is None:
        raise AttributeError("No text_col for text_analysis")

    text_col = text_col.rstrip("\r")

    clusters_descriptions = json_java.get("clustersDescriptions", [])

    cluster_words_amount = max(1, json_java.get("clusterWordsAmount", 10))

    other_in_general_cluster = json_java.get("otherInGeneralCluster", False)

    remove_duplicates = json_java.get("dropDuplicates", True)

    user_clusters = json_java.get("clustersAmount")

    _text_word2vec_params = text_config.get("text_word2vec_params")
    _text_word2vec_params["inputCol"] = text_col
    _w2v_params = setup_text_word2vec_params(_text_word2vec_params)

    # TODO: check why this doesn't work
    text_config["clustering"]["cluster_frequency"] \
        = json_java.get("clusterFrequency")
    _clustering_params = setup_text_clustering_params(text_config)
    _clustering_params.cluster_frequency = json_java.get("clusterFrequency")

    return TextAnalysisParams(
        text_col=text_col,
        user_clusters=user_clusters,
        clusters_descriptions=clusters_descriptions,
        cluster_words_amount=cluster_words_amount,
        other_in_general_cluster=other_in_general_cluster,
        remove_duplicates=remove_duplicates,
        num_features=_w2v_params.vectorSize,
        n_partitions=_w2v_params.numPartitions,
        word2vec_params=_w2v_params,
        clustering_params=_clustering_params,
    )


def setup_text_word2vec_params(param_dict: Dict) -> TextWord2VecParams:
    return TextWord2VecParams(**param_dict)


def setup_text_clustering_params(text_config: Dict) -> TextClusteringParams:
    return TextClusteringParams(
        min_cluster_num=text_config["clustering"].get("minClusterNum"),
        max_cluster_num=text_config["clustering"].get("maxClusterNum"),
        feature_cols=text_config["clustering"].get("featureCols"),
        prediction_col=text_config["clustering"].get("predictionCol"),
        evaluation=text_config["clustering"].get("evaluation"),
        clustering_seed=text_config["clustering"].get("clustering_seed"),
        cluster_frequency=text_config["clustering"].get("cluster_frequency"),
    )


def get_columns_from_params(data: Dict, meta_info: Dict) -> List[str]:
    """
    Get only "useful" columns from notationData and modelParams

    Parameters
    ----------
    data: Dict
        Json from java
    meta_info: Dict
        Meta info
    Returns
    -------
        columns : List[str]
            List with result columns
    """

    columns = []

    notation_parameters = setup_notation_params(data, stripped=False)
    raw_model_params = FrontPythonModelResponse(**data.get("mlKeys"))
    model_parameters = setup_model_params(raw_model_params, meta_info, stripped=False)
    [columns.append(col) for col in notation_parameters.dict().values() if col is not None]

    for param in model_parameters.modelParams:
        if param.containsNotation is False:
            pass
        elif param.paramType == 'list':
            if isinstance(param.paramValue, list):
                columns.extend(param.paramValue)
            elif isinstance(param.paramValue, str):
                columns.append(param.paramValue)
        elif param.paramType == 'str':
            columns.append(param.paramValue)
    columns = list(set(columns))
    return columns


def get_columns_from_params_lite(data: Dict) -> List[str]:
    """
        Get only "useful" columns from notationData

        Parameters
        ----------
        data: Dict
            Json from java
        Returns
        -------
            columns : List[str]
                List with result columns
        """

    columns = []
    notation_parameters = setup_notation_params(data, stripped=False)
    [columns.append(col) for col in notation_parameters.dict().values() if col is not None]

    return columns
