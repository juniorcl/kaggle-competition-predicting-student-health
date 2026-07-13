from .xgboost_tuning import tune_xgboost
from .catboost_tuning import tune_catboost
from .lightgbm_tuning import tune_lightgbm


MODEL_REGISTRY = {
    'xgboost': tune_xgboost,
    'lightgbm': tune_lightgbm,
    # 'catboost': tune_catboost,
}
