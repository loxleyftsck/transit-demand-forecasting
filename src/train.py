"""
Hyperparameter tuning (Optuna) + experiment tracking (MLflow) untuk LightGBM
forecasting entries per stasiun per jam.

- mlflow.lightgbm.autolog() aktif di awal -> setiap trial otomatis tercatat
  (params, metrics, model) ke MLflow tracking lokal (file store `./mlruns`).
- Optuna mencari num_leaves, max_depth, learning_rate, n_estimators (dengan
  early_stopping_rounds), min_data_in_leaf, feature_fraction, bagging_fraction
  dengan budget terbatas (N_TRIALS trial, default 18) pada validation set.
- Parameter terbaik disimpan ke models/best_params.json untuk dipakai Tahap 7
  (final training).

Catatan performa: pengecekan awal menunjukkan LightGBM melatih 100 pohon di
atas 6 juta baris data train hanya ~5-6 detik pada mesin ini (16 core), jadi
tuning dijalankan langsung di atas train/val PENUH tanpa perlu subsampling.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FINAL_MODEL_PATH = MODELS_DIR / "lightgbm_final.txt"

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "station_complex", "borough", "transit_mode",
]
CATEGORICAL_COLS = ["station_complex", "borough", "transit_mode"]
TARGET_COL = "entries"

N_TRIALS = 18
EARLY_STOPPING_ROUNDS = 50


def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    return train, val


def make_objective(train: pd.DataFrame, val: pd.DataFrame):
    Xtr, ytr = train[FEATURE_COLS], train[TARGET_COL]
    Xval, yval = val[FEATURE_COLS], val[TARGET_COL]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,  # wajib >0 supaya bagging_fraction aktif
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                eval_metric="mae",
                categorical_feature=CATEGORICAL_COLS,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            preds = model.predict(Xval)
            mae = mean_absolute_error(yval, preds)
            rmse = mean_squared_error(yval, preds) ** 0.5
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("best_iteration", model.best_iteration_)

        # early stopping bisa berhenti jauh sebelum n_estimators (batas atas
        # yang di-suggest Optuna) tercapai -- simpan jumlah pohon OPTIMAL yang
        # sebenarnya supaya Tahap 7 (final training tanpa validation set untuk
        # early-stop) tidak melatih pohon berlebih / overfit.
        trial.set_user_attr("best_iteration", model.best_iteration_)

        logger.info(
            "Trial %s selesai -> val_mae=%.3f val_rmse=%.3f best_iter=%s params=%s",
            trial.number, mae, rmse, model.best_iteration_, params,
        )
        return mae

    return objective


def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("mta-ridership-forecasting")
    # log_datasets=False: menghitung digest dataset (applymap) di atas 6 juta
    # baris sangat lambat dan tidak diperlukan untuk keperluan tuning ini.
    mlflow.lightgbm.autolog(log_datasets=False)

    logger.info("Load train & val set...")
    train, val = load_train_val()
    logger.info("Train: %s baris | Val: %s baris", len(train), len(val))

    objective = make_objective(train, val)
    study = optuna.create_study(direction="minimize", study_name="lightgbm-tuning")

    with mlflow.start_run(run_name="optuna_tuning_parent"):
        mlflow.log_param("n_trials", N_TRIALS)
        study.optimize(objective, n_trials=N_TRIALS)
        mlflow.log_metric("best_val_mae", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    best_params = dict(study.best_params)
    best_params["n_estimators"] = study.best_trial.user_attrs["best_iteration"]

    logger.info("=== RINGKASAN TUNING ===")
    logger.info("Trial terbaik : #%s", study.best_trial.number)
    logger.info("Best val MAE  : %.4f", study.best_value)
    logger.info("Best params (n_estimators sudah dikoreksi ke best_iteration): %s", best_params)

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params disimpan ke: %s", BEST_PARAMS_PATH)


def train_final() -> lgb.LGBMRegressor:
    """Latih ulang LightGBM dengan best params (Tahap 6) di atas train+val
    digabung (final training set), lalu simpan booster ke models/.

    Tidak ada early stopping di sini: `n_estimators` dipakai persis sesuai
    hasil koreksi best_iteration_ dari Tahap 6 (bukan batas atas pencarian
    Optuna), karena sudah tidak ada validation set terpisah untuk early-stop
    setelah val digabung ke training.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    logger.info("Best params (dari Tahap 6): %s", best_params)

    train, val = load_train_val()
    train_final_df = pd.concat([train, val], ignore_index=True)
    logger.info(
        "Final training set = train + val = %s baris (%s + %s)",
        len(train_final_df), len(train), len(val),
    )

    X = train_final_df[FEATURE_COLS]
    y = train_final_df[TARGET_COL]

    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("mta-ridership-forecasting")
    mlflow.lightgbm.autolog(log_datasets=False)

    with mlflow.start_run(run_name="final_model_train_plus_val"):
        model = lgb.LGBMRegressor(**best_params, n_jobs=-1, verbose=-1, random_state=42)
        model.fit(X, y, categorical_feature=CATEGORICAL_COLS)

    model.booster_.save_model(str(FINAL_MODEL_PATH))
    logger.info("=== RINGKASAN FINAL TRAINING ===")
    logger.info("Baris training final : %s", len(train_final_df))
    logger.info("n_estimators dipakai : %s", best_params["n_estimators"])
    logger.info("Model disimpan ke    : %s", FINAL_MODEL_PATH)
    return model


if __name__ == "__main__":
    import sys

    if "--final" in sys.argv:
        train_final()
    else:
        run()
