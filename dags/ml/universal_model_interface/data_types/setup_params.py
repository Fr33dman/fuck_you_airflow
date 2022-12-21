from typing import Dict

from ml.universal_model_interface.data_types.types.python_model_type import PythonModelSingleParam, PythonModelParams


def setup_python_model_params(model_dict: Dict) -> PythonModelParams:
    """
    Get PythonModelType from dict

    Parameters
    ----------
    model_dict : Dict
        Expecting model_dict like:
        {
           "modelName":"FACTOR_ANALYSIS",
           "importPath":"process_mining.sbersparkpm.ml.factor_analysis",
           "functionName":"FactorAnalysis",
           "methodToUse":"analysis",
           "modelParams":[
              {
                 "paramName":"target_column",
                 "paramType":"string",
                 "nestedParamType:"None"
                 "paramValue":"None"
              },
              {
                 "paramName":"type_of_target",
                 "paramType":"boolean",
                 "nestedParamType:"None"
                 "paramValue":false
              },
              {
                 "paramName":"categorical_cols",
                 "paramType":"list",
                 "nestedParamType":"string",
                 "paramValue":[

                 ]
              },
              {
                 "paramName":"date_cols",
                 "paramType":"list",
                 "nestedParamType":"string",
                 "paramValue":[

                 ]
              },
              {
                 "paramName":"extended_search",
                 "paramType":"boolean",
                 "nestedParamType:"None"
                 "paramValue":false
              },
              {
                 "paramName":"count_others",
                 "paramType":"boolean",
                 "nestedParamType:"None"
                 "paramValue":false
              }
           ]
        }
    Returns
    -------
        PythonModelParams
    """
    try:
        model_params = list(
            map(lambda x: PythonModelSingleParam(**x),
                model_dict["modelParams"])
        )
    except KeyError:
        raise AssertionError("Expected 'modelParams' in json")

    return PythonModelParams(
        modelName=model_dict["modelName"],
        modelLabel=model_dict["modelLabel"],
        modelAlias=model_dict["modelAlias"],
        modelDescription=model_dict["modelDescription"],
        outputDescription=model_dict["outputDescription"],
        importPath=model_dict["importPath"],
        functionName=model_dict["functionName"],
        methodToUse=model_dict["methodToUse"],
        modelParams=model_params
    )

