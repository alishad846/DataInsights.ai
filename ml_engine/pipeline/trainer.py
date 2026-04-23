import pandas as pd
import numpy as np
import os
import json
import logging
from filelock import FileLock
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer

logger = logging.getLogger("system_logger")


def get_smart_target(df, schema):
    """
    Detects the target column based on priority:
    1. Profit 2. Sales/Revenue 3. Last numeric column
    """
    if "profit_column" in schema and schema["profit_column"] in df.columns:
        return schema["profit_column"]
    if "sales_column" in schema and schema["sales_column"] in df.columns:
        return schema["sales_column"]

    # Fallback to last numeric column
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        return numeric_cols[-1]
    return None


def detect_problem_type(df, target_col):
    if df[target_col].nunique() <= 10 or pd.api.types.is_object_dtype(df[target_col]):
        return "classification"
    return "regression"


def train_evaluate_models(dataset_dir, pipeline_mode="lightweight"):
    """
    Smart training logic:
      - 'bi_only': skip training entirely (< 5,000 rows)
      - 'lightweight': RandomForest only (5,000–100,000 rows)
      - 'full': RandomForest + XGBoost (> 100,000 rows)
    """
    engineered_path = os.path.join(dataset_dir, "engineered_train_data.csv")
    schema_path = os.path.join(dataset_dir, "schema.json")
    fi_path = os.path.join(dataset_dir, "feature_importance.json")
    metrics_path = os.path.join(dataset_dir, "model_metrics.json")

    # --- bi_only mode: skip training, write minimal artifacts ---
    if pipeline_mode == "bi_only":
        logger.info("Pipeline mode = bi_only. Skipping ML training.")
        _write_skipped_artifact(fi_path, "Skipped: bi_only mode (< 5,000 rows)")
        _write_skipped_artifact(
            metrics_path, "Skipped: bi_only mode (< 5,000 rows)", is_metrics=True
        )
        return {"status": "skipped", "reason": "bi_only_mode"}

    # --- Also check if engineered file exists, fall back to train_data ---
    if not os.path.exists(engineered_path):
        engineered_path = os.path.join(dataset_dir, "train_data.csv")
        if not os.path.exists(engineered_path):
            logger.error(f"No engineered or train data found in {dataset_dir}")
            return False

    try:
        df = pd.read_csv(engineered_path)
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data for training: {str(e)}")
        return False

    target_col = get_smart_target(df, schema)
    if not target_col:
        logger.error("No valid target column found for training.")
        return False

    logger.info(f"Smart Target: {target_col} | Mode: {pipeline_mode}")

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    y = df[target_col]
    X = X.fillna(X.median())

    problem_type = detect_problem_type(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Build model list based on mode ---
    if problem_type == "classification":
        # Always use a single lightweight RF baseline
        models = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        }
        if pipeline_mode == "full":
            from xgboost import XGBClassifier

            models["XGBClassifier"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
    else:
        models = {
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        }
        if pipeline_mode == "full":
            from xgboost import XGBRegressor

            models["XGBRegressor"] = XGBRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )

    best_model = None
    best_score = -float("inf") if problem_type == "regression" else -1
    best_name = ""
    all_metrics = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
                all_metrics[name] = {
                    "accuracy": score,
                    "f1_score": f1_score(
                        y_test, preds, average="weighted", zero_division=0
                    ),
                }
            else:
                score = r2_score(y_test, preds)
                all_metrics[name] = {
                    "r2_score": score,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                }

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        except Exception as e:
            logger.warning(f"Model {name} failed: {str(e)}")

    if best_model is None:
        logger.error("All models failed.")
        return False

    # --- Feature Importance ---
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = dict(zip(X.columns, importances.tolist()))
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    with FileLock(fi_path + ".lock"):
        with open(fi_path, "w") as f:
            json.dump(
                {
                    "target": target_col,
                    "model": best_name,
                    "importance": feature_importance,
                },
                f,
                indent=4,
            )

    # --- Save best model ---
    joblib.dump(best_model, os.path.join(dataset_dir, "best_model.joblib"))

    # --- Metrics artifact ---
    final_metrics = {
        "problem_type": problem_type,
        "target_column": target_col,
        "best_model": best_name,
        "best_score": best_score,
        "pipeline_mode": pipeline_mode,
        "all_models": all_metrics,
    }
    with FileLock(metrics_path + ".lock"):
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

    logger.info(
        f"Training complete. Mode={pipeline_mode} | Best={best_name} ({best_score:.4f})"
    )

    return {"status": "success", "best_model": best_name, "score": best_score}


def _write_skipped_artifact(path, reason, is_metrics=False):
    payload = {"status": "skipped", "reason": reason}
    if is_metrics:
        payload.update({"problem_type": None, "best_model": None, "best_score": None})
    with FileLock(path + ".lock"):
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = train_evaluate_models(sys.argv[1])
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "Missing argument: dataset_dir"}))
