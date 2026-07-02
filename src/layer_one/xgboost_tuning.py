import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90, n_splits: int = 5) -> None:

    logger = setup_model_logger('xgboost')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    num_class = len(np.unique(y_train))

    folds = []
    
    for train_idx, valid_idx in cv.split(X_train, y_train):
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

        train_dmatrix = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        valid_dmatrix = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        
        folds.append((train_dmatrix, valid_dmatrix, y_va.to_numpy()))


    def objective(trial, folds):
        
        scores = []

        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "num_class": num_class,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5, log=True),
            "alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)

        for fold, (train_dmat, valid_dmat, y_valid_fold) in enumerate(folds):

            model = xgb.train(
                params=params,
                dtrain=train_dmat,
                num_boost_round=n_estimators,
                evals=[(valid_dmat, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            proba = model.predict(valid_dmat)

            score = log_loss(y_valid_fold, proba)
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)


    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, folds), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    logger.info(f"Best Log Loss: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    best_params = study.best_params.copy()
    n_estimators_final = best_params.pop("n_estimators")

    pipe_tuned = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
        enable_categorical=True,
        tree_method="hist",
        n_estimators=n_estimators_final,
        **best_params
    ).fit(X_train, y_train)

    dump_pickle(pipe_tuned, model_path)