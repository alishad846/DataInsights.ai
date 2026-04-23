import pandas as pd
import numpy as np
import os
import json
import logging
from filelock import FileLock
import joblib

logger = logging.getLogger("system_logger")


def extract_datetime_features(df, schema):
    """
    Extracts year, month, day, dayofweek from the detected date column.
    """
    date_col = schema.get("date_column")
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[f"{date_col}_year"] = df[date_col].dt.year
            df[f"{date_col}_month"] = df[date_col].dt.month
            df[f"{date_col}_day"] = df[date_col].dt.day
            df[f"{date_col}_dow"] = df[date_col].dt.dayofweek

            # Drop original date col for modeling, but keep it for analytics?
            # We'll keep it, models can just ignore datetime dtypes.
        except Exception as e:
            logger.warning(
                f"Failed to extract datetime features for {date_col}: {str(e)}"
            )
    return df


def detect_relationships(df):
    """
    Auto-detects implicit formulas like sales = price * quantity by generating
    combinations and checking for near-1.0 correlations.
    Returns a list of detected relationships.
    """
    relationships = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 3:
        return relationships

    # Limit combinations if too many columns to avoid performance hits
    if len(numeric_cols) > 20:
        numeric_cols = numeric_cols[:20]

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]

            # Avoid division by zero issues, add small epsilon
            comb_mul = df[col1] * df[col2]
            comb_add = df[col1] + df[col2]

            for target in numeric_cols:
                if target in [col1, col2]:
                    continue

                target_series = df[target]

                # Check multiplication correlation
                if len(comb_mul) > 0 and len(target_series) > 0:
                    corr_mul = comb_mul.corr(target_series)
                    if pd.notnull(corr_mul) and abs(corr_mul) > 0.98:
                        relationships.append(
                            {
                                "target": target,
                                "formula": f"{col1} * {col2}",
                                "type": "multiplication",
                                "correlation": round(corr_mul, 4),
                            }
                        )

                # Check addition correlation
                if len(comb_add) > 0 and len(target_series) > 0:
                    corr_add = comb_add.corr(target_series)
                    if pd.notnull(corr_add) and abs(corr_add) > 0.98:
                        relationships.append(
                            {
                                "target": target,
                                "formula": f"{col1} + {col2}",
                                "type": "addition",
                                "correlation": round(corr_add, 4),
                            }
                        )

    return relationships


def encode_categorical_features(df, dataset_dir):
    """
    One-hot or label encodes categorical variables dynamically.
    Skips high-cardinality ID columns (>50% unique values).
    Saves the encoders to the dataset directory artifact.
    """
    from sklearn.preprocessing import LabelEncoder

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    n_rows = len(df)

    for col in categorical_cols:
        # Skip high-cardinality identifier columns (>50% unique)
        n_unique = df[col].nunique()
        if n_unique > n_rows * 0.5:
            logger.info(
                f"Skipping high-cardinality column '{col}' ({n_unique} unique / {n_rows} rows)"
            )
            df = df.drop(columns=[col])
            continue

        # Skip obvious ID columns
        col_lower = str(col).lower()
        if col_lower.endswith("_id") or col_lower == "id":
            logger.info(f"Skipping ID column '{col}'")
            df = df.drop(columns=[col])
            continue

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    if encoders:
        encoder_path = os.path.join(dataset_dir, "label_encoders.joblib")
        joblib.dump(encoders, encoder_path)
        logger.info(
            f"Saved label encoders for {len(encoders)} columns to {encoder_path}"
        )

    return df, list(encoders.keys())


def scale_numerical_features(df, dataset_dir, exclude_cols=None):
    """
    Scales numerical features using StandardScaler.
    Excludes specified columns safely.
    """
    from sklearn.preprocessing import StandardScaler

    exclude_cols = exclude_cols or []
    all_numeric = df.select_dtypes(include=["number"]).columns.tolist()

    numeric_cols = []
    for col in all_numeric:
        col_lower = str(col).lower()
        if col in exclude_cols:
            continue
        if col_lower.endswith("_id") or col_lower == "id":
            continue
        numeric_cols.append(col)

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        scaler_path = os.path.join(dataset_dir, "standard_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaled {len(numeric_cols)} numerical features.")

    return df, numeric_cols


def derive_features(df, schema):
    """
    Derives explicit features based on schema logic (e.g. sales = price * qty).
    """
    price_col = schema.get("price_column")
    qty_col = schema.get("quantity_column")
    sales_col = schema.get("sales_column")
    derived_cols = []

    if price_col and qty_col and price_col in df.columns and qty_col in df.columns:
        if not sales_col or sales_col not in df.columns:
            if pd.api.types.is_numeric_dtype(
                df[price_col]
            ) and pd.api.types.is_numeric_dtype(df[qty_col]):
                df["derived_sales"] = df[price_col].clip(lower=0) * df[qty_col].clip(
                    lower=0
                )
                logger.info("Derived 'derived_sales' feature from price and quantity.")
                schema["sales_column"] = "derived_sales"
                derived_cols.append("derived_sales")

    return df, derived_cols


def engineer_features(dataset_dir, use_scaler=False):
    """
    Main entry for feature engineering: Dates, Encodings, and Relationship Detection.
    - use_scaler: Set True ONLY for linear models. Tree-based models (RF, XGBoost) don't benefit.
    """
    train_path = os.path.join(dataset_dir, "train_data.csv")
    schema_path = os.path.join(dataset_dir, "schema.json")

    try:
        df = pd.read_csv(train_path)
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data for feature engineering: {str(e)}")
        return False

    logger.info(f"Starting feature engineering for {dataset_dir}")

    # 1. Datetime extraction
    df = extract_datetime_features(df, schema)

    # 2. Derive Explicit Features
    df, derived_cols = derive_features(df, schema)

    # 3. Relationship Detection
    relationships = detect_relationships(df)
    if relationships:
        logger.info(f"Detected {len(relationships)} implicit relationships.")
        rel_path = os.path.join(dataset_dir, "detected_relationships.json")
        with FileLock(rel_path + ".lock"):
            with open(rel_path, "w") as f:
                json.dump(relationships, f, indent=4)

    # 4. Encoding
    df, encoded_cols = encode_categorical_features(df, dataset_dir)

    # 5. Scaling: only for linear models, skip for tree-based pipeline
    scaled_cols = []
    if use_scaler:
        df, scaled_cols = scale_numerical_features(
            df, dataset_dir, exclude_cols=encoded_cols
        )
        logger.info(
            f"StandardScaler applied to {len(scaled_cols)} columns (linear model mode)."
        )
    else:
        logger.info("StandardScaler skipped (tree-based model mode).")

    # 6. Save Feature Metadata
    metadata = {
        "derived_features": derived_cols,
        "encoded_columns": encoded_cols,
        "scaled_columns": scaled_cols,
        "scaler_applied": use_scaler,
    }
    metadata_path = os.path.join(dataset_dir, "feature_metadata.json")
    with FileLock(metadata_path + ".lock"):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # Save the engineered dataset ready for training (overwrite train_data or save as engineered_data)
    engineered_path = os.path.join(dataset_dir, "engineered_train_data.csv")
    df.to_csv(engineered_path, index=False)

    logger.info(
        f"Feature engineering complete. Derived: {len(derived_cols)} | Encoded: {len(encoded_cols)} | Scaled: {len(scaled_cols)}"
    )

    return {
        "status": "success",
        "engineered_train": engineered_path,
        "relationships_detected": len(relationships),
        "encoded_features": len(encoded_cols),
        "derived_features": len(derived_cols),
        "scaled_features": len(scaled_cols),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        result = engineer_features(dataset_dir)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "Missing argument: dataset_dir"}))
