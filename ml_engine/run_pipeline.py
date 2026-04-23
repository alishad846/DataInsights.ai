import os
import sys
import json
import logging
import time
import datetime
import argparse
import pandas as pd

from pipeline.validator import validate_dataset
from pipeline.schema_manager import process_schema
from pipeline.cleaner import clean_and_sample_dataset
from pipeline.feature_engineer import engineer_features
from pipeline.trainer import train_evaluate_models
from pipeline.forecaster import generate_forecast
from pipeline.bi_engine import run_bi_engine
from pipeline.metric_engine import generate_metric_definitions
from pipeline.insight_engine import generate_insights
from pipeline.dashboard import generate_dashboard_config

# Configure base logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("system_logger")

EXPECTED_ARTIFACTS = [
    "dataset_metadata.json",
    "schema.json",
    "profile_report.json",
    "metrics.json",
    "feature_importance.json",
    "kpi_summary.json",
    "forecast.json",
    "dashboard_config.json",
    "metrics_definition.json",
    "insights.json",
    "model_metrics.json",
]

# ─── Size Thresholds ─────────────────────────────────────────────────────────
SMALL_THRESHOLD = 5_000  # rows < 5k  → bi_only (skip ML training)
MEDIUM_THRESHOLD = 100_000  # rows < 100k → lightweight RF only
# rows >= 100k → full (RF + XGBoost)


def _timed_stage(name, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and emit a stage timing log."""
    print(f"[STAGE-START] {name}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    elapsed_ms = int(elapsed * 1000)
    print(f"[STAGE-END] {name} duration={elapsed_ms}ms")
    logger.info(f"[STAGE-END] {name} duration={elapsed_ms}ms")
    sys.stdout.flush()
    return result, elapsed


def run_pipeline(file_path, user_id="default_user", dataset_id_input=None):
    """
    Optimized DataInsights.ai ML Pipeline.
    Changes vs previous version:
      • Smart pipeline mode based on row count (bi_only / lightweight / full)
      • Dataset loaded ONCE in run_pipeline and passed down via dataset_dir
      • StandardScaler skipped for tree-based models
      • Forecasting only runs when date + target columns exist
      • Stage-level timing logged to system.log
      • In-memory caching handled in query_engine (separate)
    """
    overall_start = time.time()
    print(f"\n[PIPELINE-START] dataset_id={dataset_id_input}")
    logger.info(f"[PIPELINE-START] dataset_id={dataset_id_input}")

    result_payload = {
        "status": "failed",
        "dataset_id": dataset_id_input,
        "error": "",
        "data_quality_score": None,
        "artifacts_generated": [],
        "pipeline_mode": None,
    }

    # ── 1. Validation & Setup ───────────────────────────────────────────────
    val_res, val_time = _timed_stage(
        "validator",
        validate_dataset,
        file_path,
        base_dir=os.path.dirname(os.path.abspath(__file__)),
        user_id=user_id,
        dataset_id=dataset_id_input,
    )

    if val_res.get("status") == "error":
        result_payload["error"] = val_res.get("message")
        return result_payload

    dataset_id = val_res["dataset_id"]
    dataset_dir = val_res["dataset_dir"]
    row_count = val_res.get("rows", 0)
    result_payload["dataset_id"] = dataset_id

    # ── Determine pipeline mode ─────────────────────────────────────────────
    row_count = int(row_count) if row_count else 0
    if row_count < SMALL_THRESHOLD:
        pipeline_mode = "bi_only"
    elif row_count < MEDIUM_THRESHOLD:
        pipeline_mode = "lightweight"
    else:
        pipeline_mode = "full"

    result_payload["pipeline_mode"] = pipeline_mode
    logger.info(
        f"[{dataset_id}] Dataset size: {row_count} rows → mode: {pipeline_mode}"
    )

    training_time = 0
    forecast_time = 0
    artifact_status = "success"

    try:
        # ── 2. Schema & Profiling ───────────────────────────────────────────
        schema, _ = _timed_stage(
            "schema_manager", process_schema, val_res["raw_path"], dataset_dir
        )
        if not schema:
            raise Exception("Schema detection failed")
        result_payload["artifacts_generated"].extend(
            ["schema.json", "profile_report.json"]
        )

        # ── 3. Data Cleaning ────────────────────────────────────────────────
        clean_res, _ = _timed_stage("cleaner", clean_and_sample_dataset, dataset_dir)
        if not clean_res:
            raise Exception("Data cleaning failed")
        result_payload["data_quality_score"] = clean_res["quality_score"]

        # ── 4. Feature Engineering (scaler only for linear models) ──────────
        # In lightweight/full mode we use tree-based RF as baseline → no scaler
        feat_res, _ = _timed_stage(
            "feature_engineer",
            engineer_features,
            dataset_dir,
            use_scaler=False,  # Tree models don't need scaling
        )
        if not feat_res:
            raise Exception("Feature engineering failed")

        # ── 5. Training (smart mode) ────────────────────────────────────────
        train_fn = lambda: train_evaluate_models(
            dataset_dir, pipeline_mode=pipeline_mode
        )
        train_res, training_time = _timed_stage("trainer", train_fn)
        if train_res:
            result_payload["artifacts_generated"].append("feature_importance.json")

        # ── 6. Forecasting (guardrail: only if date + target present) ───────
        has_datetime = bool(schema.get("date_column"))
        has_target = bool(schema.get("sales_column") or schema.get("profit_column"))

        if has_datetime and has_target:
            fc_res, forecast_time = _timed_stage(
                "forecaster", generate_forecast, dataset_dir
            )
        else:
            logger.info(
                f"[{dataset_id}] Forecasting skipped — missing datetime or target column."
            )
            forecast_path = os.path.join(dataset_dir, "forecast.json")
            from filelock import FileLock

            with FileLock(forecast_path + ".lock"):
                with open(forecast_path, "w") as fh:
                    json.dump(
                        {"status": "skipped", "reason": "missing_datetime_or_target"},
                        fh,
                        indent=4,
                    )
            fc_res = {"status": "skipped"}
            forecast_time = 0.0

        result_payload["artifacts_generated"].append("forecast.json")

        # ── 7. BI Engine ────────────────────────────────────────────────────
        bi_res, _ = _timed_stage("bi_engine", run_bi_engine, dataset_dir)
        if bi_res:
            result_payload["artifacts_generated"].append("kpi_summary.json")

        # ── 8. Metric Engine ────────────────────────────────────────────────
        metric_res, _ = _timed_stage(
            "metric_engine", generate_metric_definitions, dataset_dir
        )
        if metric_res:
            result_payload["artifacts_generated"].extend(
                ["metrics.json", "metrics_definition.json"]
            )

        # ── 9. Insight Engine ───────────────────────────────────────────────
        insight_res, _ = _timed_stage("insight_engine", generate_insights, dataset_dir)
        if insight_res:
            result_payload["artifacts_generated"].append("insights.json")

        # ── 10. Dashboard Generation ────────────────────────────────────────
        dash_res, _ = _timed_stage("dashboard", generate_dashboard_config, dataset_dir)
        if dash_res:
            result_payload["artifacts_generated"].append("dashboard_config.json")

        # ── 11. Artifact Contract Verification ──────────────────────────────
        missing = [
            a
            for a in EXPECTED_ARTIFACTS
            if not os.path.exists(os.path.join(dataset_dir, a))
        ]
        if missing:
            artifact_status = "partial_failure"
            result_payload["status"] = "failed"
            result_payload["error"] = (
                f"artifact_generation_failed: Missing {', '.join(missing)}"
            )
            logger.error(f"[{dataset_id}] Missing artifacts: {', '.join(missing)}")
        else:
            result_payload["status"] = "completed"

    except Exception as e:
        logger.error(f"[{dataset_id}] Pipeline failed: {str(e)}")
        result_payload["error"] = f"Pipeline execution failed: {str(e)}"
        artifact_status = "failure"

    total_time = time.time() - overall_start
    result_payload["execution_time"] = total_time
    total_time_s = int(total_time)

    print(f"[PIPELINE-END] total_duration={total_time_s}s\n")
    logger.info(f"[PIPELINE-END] total_duration={total_time_s}s")

    logger.info(
        f"PIPELINE_RUN | User: {user_id} | Dataset: {dataset_id} | Mode: {pipeline_mode} | "
        f"TotalTime: {total_time:.2f}s | TrainTime: {training_time:.2f}s | "
        f"ForecastTime: {forecast_time:.2f}s | Rows: {row_count} | ArtifactStatus: {artifact_status}"
    )

    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DataInsights.ai ML Pipeline")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the uploaded dataset file",
    )
    parser.add_argument(
        "--user_id", type=str, default="default_user", help="Organizational User ID"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=False,
        help="Optional dataset ID from Node backend",
    )
    args = parser.parse_args()

    res = run_pipeline(args.dataset_path, args.user_id, args.dataset_id)
    print(json.dumps(res, indent=4))

    if res.get("status") == "failed":
        sys.exit(1)
    sys.exit(0)
