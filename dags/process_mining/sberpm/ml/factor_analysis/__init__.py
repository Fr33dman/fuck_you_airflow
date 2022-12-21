from process_mining.sberpm.ml.factor_analysis._factor_analyze import FactorAnalysis

__all__ = ["FactorAnalysis"]


def shap_installed():
    installed = True
    try:
        import shap
    except ModuleNotFoundError:
        from warnings import catch_warnings, simplefilter, warn

        with catch_warnings():
            simplefilter("once", ImportWarning)
            installed = False
            message = (
                "sberpm.ml.factor_analysis.ShapImportance module will not be available "
                'because "shap" library is not installed'
            )
            warn(message, ImportWarning)

    return installed
