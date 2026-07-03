import optuna

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .utils.dump_model import dump_pickle
from .utils.logging_setup import setup_model_logger, setup_optuna_logger


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, model_path: str, n_trials: int = 90, n_splits: int = 5) -> None:

    logger = setup_model_logger('lightgbm')
    setup_optuna_logger(logger)


    logger.info("----- Fine Tuning -----")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    num_class = len(np.unique(y_train))

    folds = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        
        train_dataset = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        valid_dataset = lgb.Dataset(X_va, label=y_va, reference=train_dataset, free_raw_data=False)
        
        folds.append((train_dataset, valid_dataset, y_va.to_numpy()))


    def objective(trial, folds_data):
        
        scores = []

        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_class': num_class,
            'random_state': 42,
            'n_jobs': -1,
            'feature_pre_filter': False,
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        }

        for fold, (train_ds, valid_ds, y_valid_fold) in enumerate(folds_data):

            model = lgb.train(
                params=params,
                train_set=train_ds,
                num_boost_round=2000,
                valid_sets=[valid_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            proba = model.predict(valid_ds.data)

            score = log_loss(y_valid_fold, proba)
            scores.append(score)

            trial.report(np.mean(scores), step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)


    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, folds), n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    logger.info(f"Best Log Loss: {study.best_value} | Best params: {study.best_params}")


    logger.info("----- Saving Pipeline -----")

    pipe_tuned = lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        boosting_type='gbdt',
        verbosity=-1,
        n_estimators=2000,
        random_state=42,
        feature_pre_filter=False,
        **study.best_params
    ).fit(X_train, y_train)

    dump_pickle(pipe_tuned, model_path)