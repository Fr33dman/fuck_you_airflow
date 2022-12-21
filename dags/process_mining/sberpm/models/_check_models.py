from os.path import dirname, exists, join


def get_navec_model() -> str:
    """
    Returns
    -------
    Returns the path to the "navec" location of the model
    """
    path = join(dirname(__file__), "navec_model.tar")
    if exists(path):
        return path
    else:
        raise FileNotFoundError(
            f"The model 'navec_model.tar' is missing from the library. Please download the model "
            f"and place it on this path: {dirname(__file__)}"
        )


def get_PM_model() -> str:
    """
    Returns
    -------
    Returns the path to the "Process mining" location of the model
    """
    path = join(dirname(__file__), "pm_w2v.model")
    if exists(path):
        return path
    else:
        raise FileNotFoundError(
            f"The model 'pm_w2v.model' is missing from the library. Please download the model and "
            f"place it on this path: {dirname(__file__)}"
        )
