import pandas as pd
import os
import json
import logging
from filelock import FileLock
import numpy as np

logger = logging.getLogger("system_logger")


def calculate_data_quality_score(df, missing_ratio, duplicate_ratio, outlier_ratio):
    """
    Computes a Data Quality Score (0-100) based on various impairment ratios.
    Base score is 100, then apply penalties.
    """
    score = 100

    # Missing values penalty: up to 30 points
    score -= min(missing_ratio * 100 * 0.5, 30)

    # Duplicates penalty: up to 20 points
    score -= min(duplicate_ratio * 100 * 1.0, 20)

    # Outliers penalty: up to 20 points
    score -= min(outlier_ratio * 100 * 0.8, 20)

    return max(0, round(score, 2))


def detect_outliers_iqr(df):
    """
    Detects strict statistical outliers using IQR over numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    outliers = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (
        df[numeric_cols] > (Q3 + 1.5 * IQR)
    )
    total_outliers = outliers.sum().sum()
    total_numeric_cells = len(df) * len(numeric_cols)

    return total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0


def clean_and_sample_dataset(dataset_dir):
    """
    Reads the raw data, applies sampling (for training), computes data quality,
    drops heavily null/constant columns, drops duplicate rows, and imputes remaining nulls.
    """
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    raw_path = os.path.join(dataset_dir, "raw_data.csv")

    try:
        df = pd.read_csv(raw_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed cleaning setup: {str(e)}")
        return False

    initial_rows = len(df)
    initial_cols = len(df.columns)

    # 1. Calculate Pre-Clean Metrics for Data Quality Score
    total_cells = initial_rows * initial_cols
    missing_ratio = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / initial_rows if initial_rows > 0 else 0
    outlier_ratio = detect_outliers_iqr(df)

    quality_score = calculate_data_quality_score(
        df, missing_ratio, duplicate_ratio, outlier_ratio
    )
    metadata["data_quality_score"] = quality_score
    logger.info(
        f"[{metadata['dataset_id']}] Calculated Data Quality Score: {quality_score}"
    )

    # 2. Cleanup operations
    df = df.drop_duplicates()

    # Drop columns with >70% missing
    # Threshold is number of NON-NA values required to keep the column
    thresh = int(len(df) * 0.30)
    df = df.dropna(axis=1, thresh=thresh)

    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    # Impute missing numeric values via median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute missing categorical values via mode
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    # 3. Create Sampled Dataset for Model Training (if > 500k)
    sampling_applied = False
    train_df = df.copy()
    if len(train_df) > 500000:
        logger.info(
            f"[{metadata['dataset_id']}] Dataset > 500k rows. Applying 100k sampling for training."
        )
        train_df = train_df.sample(n=100000, random_state=42)
        sampling_applied = True

    metadata["sampling_applied"] = sampling_applied

    # Save cleaned metadata
    with FileLock(metadata_path + ".lock"):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # Save datasets
    cleaned_full_path = os.path.join(dataset_dir, "cleaned_data.csv")
    cleaned_train_path = os.path.join(dataset_dir, "train_data.csv")

    df.to_csv(cleaned_full_path, index=False)
    train_df.to_csv(cleaned_train_path, index=False)

    logger.info(
        f"[{metadata['dataset_id']}] Cleaning complete. Final rows: {len(df)}, Train Rows: {len(train_df)}"
    )
    return {
        "status": "success",
        "quality_score": quality_score,
        "cleaned_full": cleaned_full_path,
        "cleaned_train": cleaned_train_path,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        result = clean_and_sample_dataset(dataset_dir)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "Missing argument: dataset_dir"}))
