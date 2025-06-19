"""Train polymer property predictors using weighted MAE.

This script demonstrates a simple pipeline to predict polymer properties
from SMILES strings. RDKit is used to compute molecular descriptors and
fingerprints. The models are trained with LightGBM and hyper-parameters
are tuned via Optuna. Performance is measured using weighted mean absolute
error (wMAE) over multiple target columns.

Example:
    python polymer_prediction_wmae.py train.csv test.csv submission.csv

The script expects the CSV files to contain a column ``SMILES`` and one or
more target property columns such as ``Tg`` or ``FFV``. Predictions for the
specified targets are written to ``submission.csv``.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import Dataset, train as lgb_train, callback
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

RDLogger = Chem.rdChemReactions.RDLogger
RDLogger.DisableLog("rdApp.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def generate_features(smiles_list: Iterable[str]) -> pd.DataFrame:
    """Return descriptor and fingerprint features for ``smiles_list``."""
    desc_names = [name for name, _ in Descriptors.descList]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        feats: Dict[str, float | int] = {}
        if mol is None:
            feats.update({f"desc_{n}": np.nan for n in desc_names})
            feats.update({f"morgan_{i}": 0 for i in range(1024)})
        else:
            for name, func in Descriptors.descList:
                try:
                    feats[f"desc_{name}"] = func(mol)
                except Exception:
                    feats[f"desc_{name}"] = np.nan
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            feats.update({f"morgan_{i}": fp[i] for i in range(1024)})
        rows.append(feats)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    n_feats: int = 512,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    feature_cols = [c for c in train_df.columns if c.startswith(("desc_", "morgan_"))]
    mask = train_df[target].notnull()
    X_train = train_df.loc[mask, feature_cols]
    y_train = train_df.loc[mask, target]
    X_test = test_df[feature_cols]

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    keep = X_train.var() > 1e-8
    X_train = X_train.loc[:, keep]
    X_test = X_test.loc[:, keep]

    if len(X_train.columns) > n_feats:
        selector = SelectKBest(f_regression, k=n_feats)
        X_train = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            columns=X_train.columns[selector.get_support()],
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            selector.transform(X_test),
            columns=X_train.columns,
            index=X_test.index,
        )

    scaler = RobustScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train, X_test, y_train

# ---------------------------------------------------------------------------
# Weighted MAE
# ---------------------------------------------------------------------------

def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted MAE given ``weights`` for each sample."""
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# ---------------------------------------------------------------------------
# LightGBM with Optuna
# ---------------------------------------------------------------------------

def tune_lgbm(X: pd.DataFrame, y: pd.Series) -> Dict[str, float | int]:
    """Tune LightGBM parameters with Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
            "verbosity": -1,
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, vl_idx in kf.split(X):
            dtrain = Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
            dval = Dataset(X.iloc[vl_idx], y.iloc[vl_idx])
            model = lgb_train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[callback.early_stopping(100), callback.log_evaluation(0)],
            )
            pred = model.predict(X.iloc[vl_idx], num_iteration=model.best_iteration)
            scores.append(mean_absolute_error(y.iloc[vl_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, float | int],
) -> np.ndarray:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(X_test))
    for tr_idx, vl_idx in kf.split(X_train):
        dtrain = Dataset(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        dval = Dataset(X_train.iloc[vl_idx], y_train.iloc[vl_idx])
        model = lgb_train(
            {"objective": "regression", "metric": "mae", "verbosity": -1, **params},
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[callback.early_stopping(200), callback.log_evaluation(0)],
        )
        preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    return preds

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train polymer models with wMAE")
    parser.add_argument("train_csv", help="Path to training data")
    parser.add_argument("test_csv", help="Path to test data")
    parser.add_argument("output_csv", help="Where to store predictions")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Tg", "FFV", "Tc", "Density", "Rg"],
        help="Target columns",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[1, 1, 1, 1, 1],
        help="Weights for wMAE (one per target)",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_features = generate_features(train_df["SMILES"])
    test_features = generate_features(test_df["SMILES"])
    train_df = pd.concat([train_df, train_features], axis=1)
    test_df = pd.concat([test_df, test_features], axis=1)

    submission = test_df[["id"]].copy()
    weights = np.asarray(args.weights)
    wmae_scores = []

    for target, w in zip(args.targets, weights):
        X_tr, X_te, y_tr = prepare_features(train_df, test_df, target)
        best_params = tune_lgbm(X_tr, y_tr)
        preds = fit_predict(X_tr, X_te, y_tr, best_params)
        submission[target] = preds
        if target in train_df.columns:
            oof_preds = fit_predict(X_tr, X_tr, y_tr, best_params)
            score = mean_absolute_error(y_tr, oof_preds)
            wmae_scores.append((w, score))

    if wmae_scores:
        score = weighted_mae(
            np.array([s for _, s in wmae_scores]),
            np.zeros(len(wmae_scores)),
            np.array([w for w, _ in wmae_scores]),
        )
        print(f"wMAE (training): {score:.4f}")

    submission.to_csv(args.output_csv, index=False)
    print(f"Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()
