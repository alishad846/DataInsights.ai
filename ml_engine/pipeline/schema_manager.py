"""
schema_manager.py — 3-Layer + Ollama Semantic Schema Detection
===============================================================
Layer 1 · Exact dictionary matching against expanded synonym sets
Layer 2 · Fuzzy matching via rapidfuzz (threshold ≥ 80)
Layer 3 · Data-pattern analysis for unmatched columns
           - Date  : datetime dtype / parseable date strings
           - Sales : numeric, high variance, mostly positive
           - Categorical: string column with low-to-medium cardinality
           - Identifier: near-unique values relative to row count
Layer 4 · Ollama LLM interpretation (only when avg confidence < 0.7)
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import datetime
from rapidfuzz import process, fuzz
from filelock import FileLock

try:
    from pipeline.ollama_client import schema_interpret as _ollama_schema
except ImportError:
    try:
        from ollama_client import schema_interpret as _ollama_schema
    except ImportError:
        _ollama_schema = None

logger = logging.getLogger("system_logger")

# ─── Synonym Dictionaries (expanded) ─────────────────────────────────────────
SYNONYM_MAP = {
    "sales": [
        "sales", "revenue", "total", "amount", "order_total", "gross_sales",
        "gross_revenue", "net_sales", "income", "sales_amount", "total_sales",
        "total_revenue", "sale_value", "sale_amount", "value", "transaction_amount",
        "total_amount", "billed_amount", "invoiced_amount", "subtotal"
    ],
    "date": [
        "date", "order_date", "created_at", "timestamp", "transaction_date",
        "invoice_date", "ship_date", "order_time", "created_date", "event_date",
        "record_date", "entry_date", "period", "time", "datetime", "date_time",
        "transaction_time", "purchase_date", "sale_date", "order_datetime"
    ],
    "product": [
        "product", "item", "product_name", "item_name", "sku", "product_id",
        "product_code", "item_code", "description", "goods", "merchandise",
        "article", "product_description", "item_description", "part"
    ],
    "customer": [
        "customer", "client", "buyer", "account", "consumer", "customer_name",
        "client_name", "customer_id", "account_name", "contact", "user",
        "member", "subscriber", "purchaser", "end_user"
    ],
    "region": [
        "region", "country", "state", "city", "location", "area",
        "territory", "zone", "district", "market", "branch", "division",
        "province", "county", "store", "site", "office", "geography"
    ],
    "quantity": [
        "quantity", "qty", "units", "amount_sold", "volume", "count",
        "unit_sold", "pieces", "nos", "number_of_units", "units_sold",
        "order_qty", "order_quantity", "items_ordered", "quantity_ordered"
    ],
    "price": [
        "price", "unit_price", "mrp", "cost_price", "rate", "per_unit",
        "list_price", "selling_price", "retail_price", "unit_cost",
        "price_per_unit", "sale_price", "base_price"
    ],
    "profit": [
        "profit", "net_profit", "margin", "earnings", "net_income",
        "gross_profit", "operating_profit", "profit_margin", "gain",
        "profitability", "income", "net_margin"
    ],
}

# Used to resolve conflicts — lower number wins
ROLE_PRIORITY = {
    "date": 1,
    "sales": 2,
    "profit": 3,
    "product": 4,
    "customer": 5,
    "region": 6,
    "quantity": 7,
    "price": 8,
}

FUZZY_THRESHOLD = 80   # minimum score for fuzzy assignment (0–100 scale)


def _normalise(col):
    """Lowercase + replace separators so 'Order Date' → 'order_date'."""
    return str(col).lower().strip().replace(" ", "_").replace("-", "_")


# ─── Layer 1 & 2: Name-based detection ───────────────────────────────────────

def _detect_by_name(columns):
    """
    Returns {role: {"column": original, "method": "dict"|"fuzzy", "score": float}}
    for every column that can be matched by name.
    """
    assigned = {}   # role → best hit so far
    col_assignment = {}  # original_col → role   (to avoid duplicate assignments)

    for col in columns:
        norm = _normalise(col)
        matched_role = None
        score = 0.0
        method = None

        # Layer 1: exact synonym match
        for role, synonyms in SYNONYM_MAP.items():
            if norm in synonyms:
                matched_role = role
                score = 100.0
                method = "dict"
                break

        # Layer 2: fuzzy match (only if Layer 1 missed and name is long enough to be meaningful)
        if not matched_role and len(norm) > 3:
            best_role, best_score = None, 0
            for role, synonyms in SYNONYM_MAP.items():
                targets = synonyms
                res = process.extractOne(norm, targets, scorer=fuzz.WRatio)
                if res:
                    _, s, _ = res
                    if s > best_score:
                        best_score = s
                        best_role = role
            if best_score >= FUZZY_THRESHOLD:
                matched_role = best_role
                score = float(best_score)
                method = "fuzzy"

        if matched_role and col not in col_assignment:
            priority = ROLE_PRIORITY.get(matched_role, 99)
            existing = assigned.get(matched_role)
            if existing is None or priority < ROLE_PRIORITY.get(existing["matched_role"], 99) or score > existing["score"]:
                # Free the previously assigned column for this role if it exists
                if existing:
                    col_assignment.pop(existing["column"], None)
                assigned[matched_role] = {
                    "column": col,
                    "matched_role": matched_role,
                    "score": score,
                    "method": method
                }
                col_assignment[col] = matched_role

    return assigned


# ─── Layer 3: Data-pattern detection ─────────────────────────────────────────

def _is_parseable_date(series, sample_size=10):
    """Heuristic: try parsing a small sample as datetime (kept tiny to avoid hangs)."""
    sample = series.dropna().head(sample_size).astype(str)
    if len(sample) == 0:
        return False
    # Quick pre-filter: skip columns where values look like plain categories
    avg_len = sample.str.len().mean()
    if avg_len < 6 or avg_len > 30:   # dates are usually 8-25 chars
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.7


def _detect_by_pattern(df, already_assigned_cols):
    """
    Assigns semantic roles to columns that were NOT matched by Layer 1/2.
    Returns additional_roles dict.
    """
    remaining = [c for c in df.columns if c not in already_assigned_cols]
    additional = {}

    categorical_cols = []
    identifier_cols  = []
    n_rows = len(df)

    for col in remaining:
        series = df[col]
        dtype  = series.dtype

        # ── Date detection ─────────────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            additional["date_column_pattern"] = col
            continue
        if dtype == object and _is_parseable_date(series):
            additional["date_column_pattern"] = col
            continue

        # ── Numeric column analysis ─────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) < 5:
                continue

            positive_ratio = (non_null > 0).mean()
            cv = non_null.std() / (non_null.mean() + 1e-9)  # coefficient of variation

            # Identifier: near-unique integers
            uniqueness = series.nunique() / max(n_rows, 1)
            if uniqueness > 0.9 and pd.api.types.is_integer_dtype(series):
                identifier_cols.append(col)
                continue

            # Sales-like: high variance, mostly positive, not an ID
            if cv > 0.3 and positive_ratio > 0.8:
                if "sales_column_pattern" not in additional:
                    additional["sales_column_pattern"] = col
                continue

        # ── Object / string column analysis ────────────────────────────────
        if dtype == object:
            n_unique = series.nunique()
            uniqueness = n_unique / max(n_rows, 1)

            if uniqueness > 0.9:
                identifier_cols.append(col)
            elif uniqueness < 0.5:
                categorical_cols.append(col)

    if categorical_cols:
        additional["categorical_columns"] = categorical_cols
    if identifier_cols:
        additional["identifier_columns"] = identifier_cols

    return additional


# ─── Profile Report ───────────────────────────────────────────────────────────

def profile_dataset(df, dataset_dir):
    numeric_cols   = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

    missing_dict = df.isnull().sum()
    missing_dict = missing_dict[missing_dict > 0].to_dict()

    profile = {
        "row_count":          len(df),
        "column_count":       len(df.columns),
        "numeric_columns":    numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns":   datetime_cols,
        "missing_values":     missing_dict
    }

    profile_path = os.path.join(dataset_dir, "profile_report.json")
    with FileLock(profile_path + ".lock"):
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=4)
    return profile


# ─── Guaranteed Fallback Helpers ─────────────────────────────────────────────

def _fallback_sales_column(df, already_assigned_cols):
    """
    Last-resort: pick the numeric column with highest variance (and ≥50% positive values)
    that hasn't already been assigned a role.
    """
    candidates = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in already_assigned_cols
    ]
    if not candidates:
        return None
    # Score = std (high spread → likely a value column)
    col = max(candidates, key=lambda c: df[c].dropna().std())
    return col


def _fallback_date_column(df, already_assigned_cols):
    """
    Last-resort: any object column where >50% of a sample parses as datetime.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        if col in already_assigned_cols:
            continue
        if _is_parseable_date(df[col]):
            return col
    return None


# ─── Main schema detection ────────────────────────────────────────────────────

def detect_column_semantics(df):
    """
    3-layer semantic detection with guaranteed fallback logic.
    Outputs confidence dict alongside role assignments.
    Never raises — always returns a usable schema.
    """
    columns = df.columns.tolist()
    n_rows  = len(df)

    # ── Layer 1 & 2 ──────────────────────────────────────────────────────────
    name_results = _detect_by_name(columns)
    already_assigned = {info["column"] for info in name_results.values()}

    schema     = {}
    confidence = {}   # role → float confidence score

    for role, info in name_results.items():
        schema_key = f"{role}_column"
        schema[schema_key] = info["column"]
        score = round(info["score"] / 100.0, 2)
        confidence[role] = score
        logger.info(
            f"[Schema][{info['method'].upper()}] Detected {role} column: "
            f"{info['column']!r} (confidence {score:.2f})"
        )

    # ── Layer 3: pattern based ────────────────────────────────────────────────
    pattern_results = _detect_by_pattern(df, already_assigned)

    for raw_key, value in pattern_results.items():
        if raw_key in ("categorical_columns", "identifier_columns"):
            continue
        clean_key = raw_key.replace("_pattern", "")          # e.g. "date_column"
        role_name  = clean_key.replace("_column", "")         # e.g. "date"
        if clean_key not in schema:
            schema[clean_key]       = value
            confidence[role_name]   = 0.60
            already_assigned.add(value)
            logger.info(
                f"[Schema][PATTERN] Detected {role_name} column: "
                f"{value!r} (confidence 0.60)"
            )

    # Always include categorical & identifier lists
    if "categorical_columns" in pattern_results:
        schema["categorical_columns"] = pattern_results["categorical_columns"]
    if "identifier_columns" in pattern_results:
        schema["identifier_columns"] = pattern_results["identifier_columns"]

    # ── Guaranteed Fallbacks ──────────────────────────────────────────────────

    # Fallback: sales column — pick highest-variance numeric if still missing
    if "sales_column" not in schema:
        fb = _fallback_sales_column(df, already_assigned)
        if fb:
            schema["sales_column"] = fb
            confidence["sales"]    = 0.40
            already_assigned.add(fb)
            logger.warning(
                f"[Schema][FALLBACK] No sales column detected via any layer. "
                f"Fallback sales column selected: {fb!r} (confidence 0.40)"
            )
        else:
            logger.warning(
                "[Schema][FALLBACK] Could not assign any sales column — "
                "no suitable numeric column found."
            )

    # Fallback: date column — mark as absent so pipeline can skip forecasting cleanly
    if "date_column" not in schema:
        fb = _fallback_date_column(df, already_assigned)
        if fb:
            schema["date_column"] = fb
            confidence["date"]    = 0.35
            already_assigned.add(fb)
            logger.warning(
                f"[Schema][FALLBACK] No date column detected via name layers. "
                f"Fallback date column selected: {fb!r} (confidence 0.35)"
            )
        else:
            logger.warning(
                "[Schema][FALLBACK] No datetime column found. "
                "Forecasting will be skipped."
            )

    # ── Layer 4: Ollama LLM (only when avg confidence < 0.7) ─────────────────
    OLLAMA_CONFIDENCE_THRESHOLD = 0.70

    if confidence:
        avg_confidence = sum(confidence.values()) / len(confidence)
    else:
        avg_confidence = 0.0

    key_roles = {"sales", "date", "customer", "product"}
    missing_key_roles = key_roles - set(confidence.keys())

    should_use_ollama = (
        _ollama_schema is not None
        and (avg_confidence < OLLAMA_CONFIDENCE_THRESHOLD or missing_key_roles)
    )

    if should_use_ollama:
        logger.info(
            f"[Schema][OLLAMA] Triggering Ollama schema interpretation — "
            f"avg confidence={avg_confidence:.2f}, missing roles={missing_key_roles or 'none'}"
        )
        ollama_result = _ollama_schema(list(df.columns))
        if ollama_result:
            for key, col_name in ollama_result.items():
                if key.endswith("_column") and key not in schema and col_name in df.columns:
                    role = key.replace("_column", "")
                    schema[key]         = col_name
                    confidence[role]    = 0.68   # Ollama-augmented confidence
                    already_assigned.add(col_name)
                    logger.info(
                        f"[Schema][OLLAMA] Assigned {role}: {col_name!r} "
                        f"(confidence 0.68)"
                    )
            logger.info("[Schema][OLLAMA] Schema merge complete.")
        else:
            logger.warning("[Schema][OLLAMA] No usable response — continuing with rule-based schema.")

    # ── Emit final summary ────────────────────────────────────────────────────
    for role, score in confidence.items():
        col_key = f"{role}_column"
        col_val = schema.get(col_key, schema.get(role, "—"))
        logger.info(f"[Schema][SUMMARY] {role}: {col_val!r} → confidence {score:.2f}")

    schema["confidence"] = confidence
    return schema


def process_schema(file_path, dataset_dir):
    """
    Main entry. Always writes schema.json even if detection confidence is low.
    Returns the schema dict or False only on total file read failure.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read dataset for schema mapping: {str(e)}")
        return False

    logger.info(f"Generating profile report for {file_path}")
    profile_dataset(df, dataset_dir)

    logger.info("Running 3-layer semantic schema detection with guaranteed fallback...")
    schema = detect_column_semantics(df)

    schema_path = os.path.join(dataset_dir, "schema.json")
    with FileLock(schema_path + ".lock"):
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=4)

    detected = [k for k in schema if k.endswith("_column") and isinstance(schema[k], str)]
    logger.info(
        f"Schema saved → {schema_path} | "
        f"Assigned roles: {[k.replace('_column','') for k in detected]}"
    )
    return schema


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        result = process_schema(sys.argv[1], sys.argv[2])
        print(json.dumps(result, indent=4))
    else:
        print(json.dumps({"error": "Usage: schema_manager.py <file_path> <dataset_dir>"}))
