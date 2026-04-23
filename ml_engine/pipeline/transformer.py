import pandas as pd
import sys
import json
import os
import argparse
import numpy as np

def transform_data(input_file, output_file, config):
    """
    Applies a specific transformation to a CSV file and saves the result.
    The transformation is applied stateless-ly on the input_file.
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Failed to read input file: {str(e)}"}))
        return

    t_type = config.get("type")
    params = config.get("params", {})
    
    try:
        if t_type == "null_fill":
            # params = { col_name: strategy }
            for col, strategy in params.items():
                if col not in df.columns: continue
                if strategy == "Fill with 0":
                    df[col] = df[col].fillna(0)
                elif strategy == "Fill with mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                elif strategy == "Fill with median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                elif strategy == "Fill with mode":
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                elif strategy == "Drop rows":
                    df = df.dropna(subset=[col])

        elif t_type == "drop_duplicates":
            strategy = params.get("strategy", "Keep first")
            if strategy == "Keep first":
                df = df.drop_duplicates(keep="first")
            elif strategy == "Keep last":
                df = df.drop_duplicates(keep="last")

        elif t_type == "type_conversion":
            # params = { col_name: target_type }
            for col, target_type in params.items():
                if col not in df.columns: continue
                try:
                    if target_type == "Integer":
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif target_type == "Float":
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    elif target_type == "String":
                        df[col] = df[col].astype(str)
                    elif target_type == "Boolean":
                        df[col] = df[col].map({'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False})
                    elif target_type == "Date":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass

        elif t_type == "outlier_handling":
            # params = { col_name: strategy }
            for col, strategy in params.items():
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if strategy == "Remove rows":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif strategy == "IQR capping":
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        elif t_type == "get_stats":
            # Just get statistics without modifying data
            null_counts = df.isnull().sum().to_dict()
            total_nulls = int(df.isnull().sum().sum())
            total_duplicates = int(df.duplicated().sum())
            print(json.dumps({
                "status": "success",
                "total_rows": len(df),
                "total_nulls": total_nulls,
                "total_duplicates": total_duplicates,
                "column_nulls": null_counts
            }))
            return

        elif t_type == "feature_eng":
            # params = { features: [ { col, type, operation, inputs, originalCol, ... } ] }
            features = params.get("features", [])
            for feat in features:
                new_col = feat.get("col") or feat.get("output") or feat.get("newCol")
                operation = feat.get("operation") or feat.get("kind") or feat.get("type")
                inputs = feat.get("inputs") or []
                orig = feat.get("originalCol")

                if not inputs and orig:
                    inputs = [orig]

                inputs = [col for col in inputs if col in df.columns]
                if not new_col or not inputs:
                    continue

                numeric_inputs = [pd.to_numeric(df[col], errors='coerce').fillna(0) for col in inputs]

                if operation in ("numeric", "normalize"):
                    source = numeric_inputs[0]
                    max_val = source.max()
                    df[new_col] = source / max_val if max_val and max_val != 0 else 0
                elif operation in ("boolean", "boolean_flag"):
                    source = df[inputs[0]]
                    df[new_col] = source.notnull() & (source.astype(str).str.strip() != "")
                    df[new_col] = df[new_col].astype(int)
                elif operation == "product":
                    result = numeric_inputs[0].copy()
                    for series in numeric_inputs[1:]:
                        result = result * series
                    df[new_col] = result
                elif operation == "sum":
                    result = numeric_inputs[0].copy()
                    for series in numeric_inputs[1:]:
                        result = result + series
                    df[new_col] = result
                elif operation == "difference":
                    first = numeric_inputs[0]
                    second = numeric_inputs[1] if len(numeric_inputs) > 1 else 0
                    df[new_col] = first - second
                elif operation == "ratio":
                    numerator = numeric_inputs[0]
                    denominator = numeric_inputs[1] if len(numeric_inputs) > 1 else 0
                    df[new_col] = np.where(denominator != 0, numerator / denominator, 0)
                elif operation == "log":
                    source = numeric_inputs[0]
                    df[new_col] = np.log1p(source.clip(lower=0))
                else:
                    # Backwards-compatible fallback for older suggestions.
                    source = numeric_inputs[0]
                    max_val = source.max()
                    if operation == "boolean" or feat.get("type") == "boolean":
                        df[new_col] = df[inputs[0]].notnull().astype(int)
                    else:
                        df[new_col] = source / max_val if max_val and max_val != 0 else 0

        else:
            print(json.dumps({"status": "error", "message": f"Unknown transformation type: {t_type}"}))
            return

        # Write output
        df.to_csv(output_file, index=False)
        print(json.dumps({"status": "success", "rows": len(df), "cols": len(df.columns)}))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True) # JSON string
    args = parser.parse_args()
    
    try:
        config_dict = json.loads(args.config)
        transform_data(args.input, args.output, config_dict)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Main error: {str(e)}"}))
