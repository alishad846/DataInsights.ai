import pandas as pd
import os
import json
import logging
import uuid
import datetime
from filelock import FileLock

# Setup basic logging locally to the module if system.log isn't configured yet
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_ROWS = 5000000
MAX_COLS = 200


def setup_dataset_versioning(
    base_dir="ml_engine", user_id="default_user", dataset_id=None
):
    """
    Creates a new dataset version folder and initializes system.log if needed.
    Multi-tenant isolation: data/users/<user_id>/<dataset_id>
    """
    user_dir = os.path.join(base_dir, "data", "users", user_id)
    logs_dir = os.path.join(base_dir, "logs")

    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Configure system logger
    log_file = os.path.join(logs_dir, "system.log")
    logger = logging.getLogger("system_logger")
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    # Use provided dataset_id or generate isolated dataset ID
    if not dataset_id:
        dataset_id = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dataset_dir = os.path.join(user_dir, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    return dataset_id, dataset_dir, logger


def validate_dataset(
    file_path, base_dir="ml_engine", user_id="default_user", dataset_id=None
):
    """
    Validates the dataset size and initial data quality to fail fast if necessary.
    Creates the isolated metadata context folder.
    """
    start_time = datetime.datetime.now()
    dataset_id, dataset_dir, logger = setup_dataset_versioning(
        base_dir, user_id, dataset_id
    )

    logger.info(
        f"Starting validation for {dataset_id} belonging to {user_id} from {file_path}"
    )

    # 1. Format check
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        error_msg = f"Invalid file format: {ext}. Only CSV and XLSX are supported."
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    try:
        if ext in [".xlsx", ".xls"]:
            # Optimization: immediately convert Excel to CSV for faster downstream I/O
            logger.info(
                f"[{dataset_id}] Converting Excel file to CSV for faster processing..."
            )
            df = pd.read_excel(file_path)
            csv_conversion_path = os.path.splitext(file_path)[0] + "_converted.csv"
            df.to_csv(csv_conversion_path, index=False)
            file_path = csv_conversion_path  # use CSV from here on
            ext = ".csv"
            logger.info(f"[{dataset_id}] Excel converted to CSV: {csv_conversion_path}")
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"[{dataset_id}] Failed to read file: {str(e)}")
        return {"status": "error", "message": f"Failed to read file: {str(e)}"}

    rows, cols = df.shape

    # 2. Duplicate columns check
    if len(df.columns) != len(set(df.columns)):
        error_msg = "Dataset contains duplicate column names."
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    # 3. Minimum row check
    if rows < 10:
        error_msg = f"Dataset is too small ({rows} rows). At least 10 rows required for reliable ML."
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    # 4. Max row/col check
    if rows > MAX_ROWS or cols > MAX_COLS:
        error_msg = (
            f"Dataset exceeds limits. Rows: {rows}/{MAX_ROWS}, Cols: {cols}/{MAX_COLS}"
        )
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    # 5. Numeric columns presence
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        error_msg = "Dataset contains no numeric columns. At least one numeric column is required for analysis."
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    # 6. Check extreme missing values
    total_cells = rows * cols
    total_missing = df.isnull().sum().sum()
    if total_cells > 0 and (total_missing / total_cells) > 0.90:
        error_msg = "Dataset is >90% empty/null. Not enough data to proceed."
        logger.error(f"[{dataset_id}] {error_msg}")
        return {"status": "error", "message": error_msg}

    # Validation Passed. Generate Initial Metadata safely with lock
    metadata = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "upload_time": start_time.isoformat(),
        "total_rows": rows,
        "total_columns": cols,
        "sampling_applied": False,
        "data_quality_score": None,  # Will be filled by cleaner
        "file_source": file_path,
    }

    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    with FileLock(metadata_path + ".lock"):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # Save a copy of the raw valid data locally to the dataset folder
    raw_path = os.path.join(dataset_dir, "raw_data.csv")
    # Use DataFrame to_csv instead of shutil.copy2 to avoid Windows file-lock issues
    df.to_csv(raw_path, index=False)

    runtime = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(
        f"[{dataset_id}] Validation complete in {runtime}s. Saved to {dataset_dir}"
    )

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "user_id": user_id,
        "dataset_dir": dataset_dir,
        "raw_path": raw_path,
        "metadata_path": metadata_path,
        "rows": rows,
        "cols": cols,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--user_id", type=str, default="default_user")
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=".")
    args = parser.parse_args()

    result = validate_dataset(
        args.file_path,
        base_dir=args.base_dir,
        user_id=args.user_id,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result))
