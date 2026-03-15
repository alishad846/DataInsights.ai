"""
query_engine.py — Intelligent Dataset Q&A Engine
==================================================
Reads pre-computed pipeline artifacts (schema.json, kpi_summary.json,
metrics.json, insights.json) and answers natural-language questions
about any uploaded dataset.

Usage (CLI):
    python query_engine.py \
        --user_id  <user_id>   \
        --dataset_id <id>      \
        --question  "what is the total sales?"

Output (stdout):
    { "answer": "...", "intent": "...", "confidence": 0.9 }
"""

import argparse
import json
import os
import sys
import re
import logging
import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger("query_engine")
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s]: %(message)s",
    stream=sys.stderr,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "users")


def resolve_dataset_dir(user_id: str, dataset_id: str) -> str:
    return os.path.normpath(
        os.path.join(BASE_DATA_DIR, user_id, dataset_id)
    )


def _load_json(path: str) -> dict | list | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
    return None


def _load_csv(path: str) -> "pd.DataFrame | None":
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Could not read CSV {path}: {e}")
    return None


# ─── Artifact Loader ──────────────────────────────────────────────────────────

class DatasetArtifacts:
    """Lazily loads all artifacts for a dataset directory."""

    def __init__(self, dataset_dir: str):
        self.dir = dataset_dir
        self._schema   = None
        self._kpi      = None
        self._metrics  = None
        self._insights = None
        self._profile  = None
        self._df       = None

    @property
    def schema(self) -> dict:
        if self._schema is None:
            self._schema = _load_json(os.path.join(self.dir, "schema.json")) or {}
        return self._schema

    @property
    def kpi(self) -> dict:
        if self._kpi is None:
            self._kpi = _load_json(os.path.join(self.dir, "kpi_summary.json")) or {}
        return self._kpi

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = _load_json(os.path.join(self.dir, "metrics.json")) or {}
        return self._metrics

    @property
    def insights(self) -> dict:
        if self._insights is None:
            self._insights = _load_json(os.path.join(self.dir, "insights.json")) or {}
        return self._insights

    @property
    def profile(self) -> dict:
        if self._profile is None:
            self._profile = _load_json(os.path.join(self.dir, "profile_report.json")) or {}
        return self._profile

    @property
    def df(self) -> "pd.DataFrame | None":
        if self._df is None:
            cleaned = os.path.join(self.dir, "cleaned_data.csv")
            raw_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.dir)))
            # Try cleaned first, then look for original upload
            self._df = _load_csv(cleaned)
        return self._df

    @property
    def available(self) -> bool:
        return bool(self.schema) or (self.df is not None)


# ─── Intent Router ───────────────────────────────────────────────────────────

INTENTS = {
    "greeting": [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreet", r"\bgood\s+(morning|afternoon|evening)\b",
        r"\bwhat can you (do|help|answer)\b", r"\bhelp\b", r"\bwhat are you\b", r"\babilities\b",
    ],
    "total_sales": [
        r"\btotal\s+sales?\b", r"\btotal\s+revenue\b", r"\bsum\s+of\s+sales?\b",
        r"\boverall\s+sales?\b", r"\bgross\s+sales?\b", r"\bhow\s+much\s+(did|were)\s+(we\s+)?sell\b",
        r"\bsales?\s+total\b",
    ],
    "total_generic": [
        r"\btotal\b", r"\bsum\s+of\b", r"\baggregate\b",
    ],
    "average": [
        r"\baverage\b", r"\bavg\b", r"\bmean\b", r"\bper\s+\w+\b", r"\btypical\b",
    ],
    "maximum": [
        r"\bmax(imum)?\b", r"\bhighest\b", r"\bbiggest\b", r"\bmost\b", r"\btop\b", r"\bbest\b",
        r"\bpeak\b", r"\bgreatest\b",
    ],
    "minimum": [
        r"\bmin(imum)?\b", r"\blowest\b", r"\bsmallest\b", r"\bworst\b", r"\bbottom\b",
        r"\bweakest\b",
    ],
    "top_n": [
        r"\btop\s+\d+\b", r"\bbest\s+\d+\b", r"\bhighest\s+\d+\b",
    ],
    "count": [
        r"\bhow\s+many\b", r"\bcount\b", r"\bnumber\s+of\b", r"\bquantity\b", r"\btotal\s+records?\b",
        r"\btotal\s+rows?\b", r"\btotal\s+entries\b",
    ],
    "region": [
        r"\bregion\b", r"\bcountry\b", r"\bcountries\b", r"\bcity\b", r"\bcities\b",
        r"\blocation\b", r"\bterritory\b", r"\bzone\b", r"\barea\b", r"\bmarket\b",
        r"\bby\s+region\b", r"\bby\s+location\b", r"\bby\s+country\b", r"\bstate\b",
    ],
    "product": [
        r"\bproduct\b", r"\bproducts\b", r"\bitem\b", r"\bitems\b", r"\bsku\b",
        r"\bbest.sell\w+\b", r"\btop.sell\w+\b", r"\bgoods\b",
    ],
    "customer": [
        r"\bcustomer\b", r"\bcustomers\b", r"\bclient\b", r"\bclients\b", r"\bbuyer\b",
        r"\bbuyers\b", r"\baccount\b",
    ],
    "profit": [
        r"\bprofit\b", r"\bprofitable\b", r"\bprofitability\b", r"\bmargin\b",
        r"\bnet\s+income\b", r"\bearnings\b",
    ],
    "loss": [
        r"\bloss\b", r"\blosses\b", r"\bnegative\b", r"\bdeficit\b", r"\bin\s+the\s+red\b",
    ],
    "trend": [
        r"\btrend\b", r"\bover\s+time\b", r"\bmonthly\b", r"\bby\s+month\b",
        r"\bquarterly\b", r"\bby\s+year\b", r"\bannual\b", r"\bhistorical\b",
        r"\btime\s+series\b", r"\bgrowth\b",
    ],
    "forecast": [
        r"\bforecast\b", r"\bpredict\b", r"\bnext\s+(month|quarter|year)\b",
        r"\bfuture\b", r"\bprojection\b", r"\bexpect\b", r"\bestimate\b",
    ],
    "anomaly": [
        r"\banomal\w+\b", r"\boutlier\b", r"\bspike\b", r"\bunusual\b",
        r"\babnormal\b", r"\bweird\b", r"\bstrange\b",
    ],
    "comparison": [
        r"\bcompar\w+\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference\s+between\b",
        r"\bwhich\s+is\s+(better|worse|higher|lower)\b",
    ],
    "insight": [
        r"\binsight\b", r"\banalysis\b", r"\banalyze\b", r"\brecommend\b",
        r"\bsuggestion\b", r"\bwhat\s+should\b",
    ],
    "summary": [
        r"\bsummary\b", r"\boverview\b", r"\bsummariz\w+\b", r"\bdescrib\w+\b",
        r"\btell\s+me\s+about\b", r"\bwhat\s+is\s+in\b", r"\bwhat\s+does\s+the\s+data\b",
        r"\babout\s+the\s+data\b",
    ],
    "columns": [
        r"\bcolumn\b", r"\bfield\b", r"\bheader\b", r"\bvariable\b",
        r"\bwhat\s+data\b", r"\bwhat\s+columns\b", r"\bwhat\s+fields\b",
        r"\bstructure\b",
    ],
    "rows": [
        r"\brow\b", r"\brecord\b", r"\bentry\b", r"\bdata\s+point\b",
        r"\bsize\s+of\b", r"\bhow\s+large\b",
    ],
    "missing": [
        r"\bmissing\b", r"\bnull\b", r"\bnan\b", r"\bempty\b", r"\bblank\b",
        r"\bincomplete\b",
    ],
}


def detect_intent(question: str) -> str:
    """Returns the best matching intent for a question."""
    q = question.lower()

    # Priority intents (checked first)
    priority_order = [
        "greeting", "top_n", "total_sales", "anomaly", "forecast",
        "trend", "region", "product", "customer", "profit", "loss",
        "columns", "rows", "missing", "summary", "insight",
        "count", "maximum", "minimum", "average", "total_generic", "comparison",
    ]

    for intent in priority_order:
        patterns = INTENTS.get(intent, [])
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "unknown"


# ─── Number Formatter ─────────────────────────────────────────────────────────

def _fmt(val) -> str:
    """Format a number nicely: 1234567 → 1,234,567.00"""
    try:
        f = float(val)
        if abs(f) >= 1_000_000:
            return f"{f:,.0f}"
        return f"{f:,.2f}"
    except Exception:
        return str(val)


def _extract_n(question: str, default: int = 5) -> int:
    """Extract the N from 'top 10 products', etc."""
    m = re.search(r"\b(?:top|best|highest|bottom|lowest|worst)\s+(\d+)\b", question.lower())
    if m:
        return int(m.group(1))
    return default


# ─── Answer Generators ────────────────────────────────────────────────────────

def _answer_greeting(question: str) -> dict:
    capabilities = (
        "Here are things I can help you with:\n\n"
        "• **Total, average, max/min** of any numeric column\n"
        "• **Top N products, regions, customers** by sales or profit\n"
        "• **Monthly/quarterly trends** over time\n"
        "• **Profit & loss** breakdown\n"
        "• **Forecasts** for next period\n"
        "• **Anomalies & outliers** in your data\n"
        "• **Dataset summary, structure & missing values**\n"
        "• **AI insights** from your data\n\n"
        "Just ask in plain English! For example:\n"
        "*\"What is the total revenue?\"* or *\"Show me top 5 products by sales.\"*"
    )
    return {"answer": capabilities, "intent": "greeting", "confidence": 1.0}


def _answer_summary(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    kpi     = artifacts.kpi
    schema  = artifacts.schema

    parts = []

    if profile:
        rows = profile.get("row_count", "?")
        cols = profile.get("column_count", "?")
        parts.append(f"**Dataset Overview:** {rows:,} rows × {cols} columns.")

        num_cols  = profile.get("numeric_columns", [])
        cat_cols  = profile.get("categorical_columns", [])
        if num_cols:
            parts.append(f"**Numeric columns:** {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}.")
        if cat_cols:
            parts.append(f"**Categorical columns:** {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}.")

        missing = profile.get("missing_values", {})
        if missing:
            mv_str = ", ".join(f"{c}: {v}" for c, v in list(missing.items())[:5])
            parts.append(f"**Missing values detected in:** {mv_str}.")
        else:
            parts.append("**No missing values** found in the dataset. ✅")

    # Key KPIs
    if kpi:
        sales_col = schema.get("sales_column")
        if sales_col and sales_col in kpi:
            s = kpi[sales_col]
            total = s.get("sum") or s.get("total")
            if total:
                parts.append(f"**Total {sales_col}:** {_fmt(total)}.")

    if not parts:
        return {"answer": "I could not find a summary for this dataset. Try waiting for processing to complete.", "intent": "summary", "confidence": 0.5}

    return {"answer": "\n\n".join(parts), "intent": "summary", "confidence": 0.9}


def _answer_columns(artifacts: DatasetArtifacts) -> dict:
    schema  = artifacts.schema
    profile = artifacts.profile

    known_roles = {
        k.replace("_column", ""): v
        for k, v in schema.items()
        if k.endswith("_column") and isinstance(v, str)
    }

    if not known_roles and not profile:
        return {"answer": "Column information is not available yet. Please ensure the dataset has been processed.", "intent": "columns", "confidence": 0.5}

    parts = []

    if known_roles:
        role_lines = "\n".join(f"  • **{role.capitalize()}** → `{col}`" for role, col in known_roles.items())
        parts.append(f"**Detected column roles:**\n{role_lines}")

    if profile:
        num_cols = profile.get("numeric_columns", [])
        cat_cols = profile.get("categorical_columns", [])
        dt_cols  = profile.get("datetime_columns", [])
        if num_cols:
            parts.append(f"**Numeric columns ({len(num_cols)}):** {', '.join(num_cols)}.")
        if cat_cols:
            parts.append(f"**Text/Category columns ({len(cat_cols)}):** {', '.join(cat_cols)}.")
        if dt_cols:
            parts.append(f"**Date columns ({len(dt_cols)}):** {', '.join(dt_cols)}.")

    return {"answer": "\n\n".join(parts), "intent": "columns", "confidence": 0.9}


def _answer_rows(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile and "row_count" in profile:
        r = profile["row_count"]
        c = profile.get("column_count", "?")
        return {"answer": f"The dataset contains **{r:,} rows** and **{c} columns**.", "intent": "rows", "confidence": 1.0}
    df = artifacts.df
    if df is not None:
        return {"answer": f"The dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**.", "intent": "rows", "confidence": 0.9}
    return {"answer": "Row count is not available yet.", "intent": "rows", "confidence": 0.4}


def _answer_missing(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile:
        missing = profile.get("missing_values", {})
        if not missing:
            return {"answer": "✅ **No missing values** were detected in your dataset.", "intent": "missing", "confidence": 1.0}
        lines = "\n".join(f"  • **{c}**: {v} missing value(s)" for c, v in missing.items())
        return {"answer": f"**Missing values detected:**\n{lines}", "intent": "missing", "confidence": 1.0}
    df = artifacts.df
    if df is not None:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return {"answer": "✅ **No missing values** were detected in your dataset.", "intent": "missing", "confidence": 0.9}
        lines = "\n".join(f"  • **{c}**: {v}" for c, v in missing.items())
        return {"answer": f"**Missing values detected:**\n{lines}", "intent": "missing", "confidence": 0.9}
    return {"answer": "Missing value information is not available.", "intent": "missing", "confidence": 0.4}


def _answer_total(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi    = artifacts.kpi

    # Determine target column from question
    target_col = _find_target_column(question, schema, artifacts.df)

    # Try KPI cache first
    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("sum") or kpi[target_col].get("total")
        if val is not None:
            return {"answer": f"The **total {target_col}** is **{_fmt(val)}**.", "intent": "total_sales" if "sales" in question.lower() or "revenue" in question.lower() else "total_generic", "confidence": 0.95}

    # Fallback: compute from CSV
    df = artifacts.df
    if df is not None and target_col and target_col in df.columns:
        val = df[target_col].sum()
        return {"answer": f"The **total {target_col}** is **{_fmt(val)}**.", "intent": "total_generic", "confidence": 0.85}

    # Try sales column directly
    sales_col = schema.get("sales_column")
    if sales_col and kpi and sales_col in kpi:
        val = kpi[sales_col].get("sum") or kpi[sales_col].get("total")
        if val is not None:
            return {"answer": f"The **total {sales_col}** is **{_fmt(val)}**.", "intent": "total_sales", "confidence": 0.9}

    if df is not None and sales_col and sales_col in df.columns:
        val = df[sales_col].sum()
        return {"answer": f"The **total {sales_col}** is **{_fmt(val)}**.", "intent": "total_sales", "confidence": 0.8}

    return {"answer": "I could not compute a total from your dataset. Please ensure the dataset has been processed successfully.", "intent": "total_generic", "confidence": 0.3}


def _answer_average(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi    = artifacts.kpi
    target_col = _find_target_column(question, schema, artifacts.df)

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("mean") or kpi[target_col].get("avg")
        if val is not None:
            return {"answer": f"The **average {target_col}** is **{_fmt(val)}**.", "intent": "average", "confidence": 0.95}

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns:
        val = df[target_col].mean()
        return {"answer": f"The **average {target_col}** is **{_fmt(val)}**.", "intent": "average", "confidence": 0.85}

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].mean()
        return {"answer": f"The **average {sales_col}** per record is **{_fmt(val)}**.", "intent": "average", "confidence": 0.75}

    return {"answer": "I could not compute an average. Please check the dataset has been processed.", "intent": "average", "confidence": 0.3}


def _answer_max(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi    = artifacts.kpi
    target_col = _find_target_column(question, schema, artifacts.df)

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("max")
        if val is not None:
            return {"answer": f"The **maximum {target_col}** is **{_fmt(val)}**.", "intent": "maximum", "confidence": 0.95}

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns:
        val = df[target_col].max()
        return {"answer": f"The **maximum {target_col}** is **{_fmt(val)}**.", "intent": "maximum", "confidence": 0.85}

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].max()
        return {"answer": f"The **highest {sales_col}** in a single record is **{_fmt(val)}**.", "intent": "maximum", "confidence": 0.75}

    return {"answer": "I could not find the maximum value.", "intent": "maximum", "confidence": 0.3}


def _answer_min(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi    = artifacts.kpi
    target_col = _find_target_column(question, schema, artifacts.df)

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("min")
        if val is not None:
            return {"answer": f"The **minimum {target_col}** is **{_fmt(val)}**.", "intent": "minimum", "confidence": 0.95}

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns:
        val = df[target_col].min()
        return {"answer": f"The **minimum {target_col}** is **{_fmt(val)}**.", "intent": "minimum", "confidence": 0.85}

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].min()
        return {"answer": f"The **lowest {sales_col}** in a single record is **{_fmt(val)}**.", "intent": "minimum", "confidence": 0.75}

    return {"answer": "I could not find the minimum value.", "intent": "minimum", "confidence": 0.3}


def _answer_count(question: str, artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile and "row_count" in profile:
        val = profile["row_count"]
        return {"answer": f"Your dataset has **{val:,} records** (rows).", "intent": "count", "confidence": 1.0}
    df = artifacts.df
    if df is not None:
        return {"answer": f"Your dataset has **{len(df):,} records** (rows).", "intent": "count", "confidence": 0.9}
    return {"answer": "Record count is not available.", "intent": "count", "confidence": 0.3}


def _answer_top_n(question: str, artifacts: DatasetArtifacts) -> dict:
    n = _extract_n(question)
    schema = artifacts.schema
    df     = artifacts.df

    # Determine groupby column (product/region/customer)
    group_col   = None
    group_label = "category"

    if re.search(r"\bproduct\b|\bitem\b|\bsku\b", question.lower()):
        group_col   = schema.get("product_column")
        group_label = "product"
    elif re.search(r"\bregion\b|\bcountry\b|\blocation\b|\barea\b|\bcity\b|\bstate\b", question.lower()):
        group_col   = schema.get("region_column")
        group_label = "region"
    elif re.search(r"\bcustomer\b|\bclient\b|\bbuyer\b", question.lower()):
        group_col   = schema.get("customer_column")
        group_label = "customer"

    # Determine value column
    val_col = _find_target_column(question, schema, df)
    if not val_col:
        val_col = schema.get("sales_column") or schema.get("profit_column")

    if df is None:
        return {"answer": "Dataset not yet available for analysis.", "intent": "top_n", "confidence": 0.3}

    # If no group column detected, try any categorical + numeric combo
    if not group_col:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        if cats:
            group_col   = cats[0]
            group_label = group_col
        if not val_col and nums:
            val_col = nums[0]

    if not group_col or not val_col or group_col not in df.columns or val_col not in df.columns:
        return {"answer": f"I could not determine a grouping or value column to compute top {n}.", "intent": "top_n", "confidence": 0.4}

    is_worst = bool(re.search(r"\bworst\b|\blowest\b|\bbottom\b", question.lower()))

    agg = df.groupby(group_col)[val_col].sum().sort_values(ascending=is_worst)
    top = agg.head(n)

    label = "bottom" if is_worst else "top"
    lines = "\n".join(f"  {i+1}. **{name}** — {_fmt(val)}" for i, (name, val) in enumerate(top.items()))
    return {
        "answer": f"**{label.capitalize()} {n} {group_label}s by {val_col}:**\n{lines}",
        "intent": "top_n",
        "confidence": 0.9
    }


def _answer_region(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df

    region_col = schema.get("region_column")
    val_col    = _find_target_column(question, schema, df) or schema.get("sales_column") or schema.get("profit_column")

    if not region_col or df is None:
        return {"answer": "Region column could not be detected in your dataset.", "intent": "region", "confidence": 0.4}

    if val_col not in df.columns:
        return {"answer": f"Could not find a suitable value column to aggregate by region.", "intent": "region", "confidence": 0.4}

    agg = df.groupby(region_col)[val_col].sum().sort_values(ascending=False)
    best  = agg.idxmax()
    worst = agg.idxmin()

    top5 = agg.head(5)
    lines = "\n".join(f"  {i+1}. **{r}** — {_fmt(v)}" for i, (r, v) in enumerate(top5.items()))

    answer = (
        f"**Regional breakdown by {val_col}:**\n{lines}\n\n"
        f"🏆 **Best region:** {best} ({_fmt(agg[best])})\n"
        f"⚠️ **Weakest region:** {worst} ({_fmt(agg[worst])})"
    )
    return {"answer": answer, "intent": "region", "confidence": 0.9}


def _answer_product(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df

    product_col = schema.get("product_column")
    val_col     = _find_target_column(question, schema, df) or schema.get("sales_column")

    if not product_col or df is None:
        return _answer_top_n(question + " by product", artifacts)

    if val_col not in df.columns:
        return {"answer": "Could not find a numeric column to rank products by.", "intent": "product", "confidence": 0.4}

    agg   = df.groupby(product_col)[val_col].sum().sort_values(ascending=False)
    best  = agg.idxmax()
    worst = agg.idxmin()

    top5 = agg.head(5)
    lines = "\n".join(f"  {i+1}. **{p}** — {_fmt(v)}" for i, (p, v) in enumerate(top5.items()))

    answer = (
        f"**Top products by {val_col}:**\n{lines}\n\n"
        f"🏆 **Best-selling:** {best} ({_fmt(agg[best])})\n"
        f"⚠️ **Lowest performing:** {worst} ({_fmt(agg[worst])})"
    )
    return {"answer": answer, "intent": "product", "confidence": 0.9}


def _answer_customer(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df

    cust_col = schema.get("customer_column")
    val_col  = _find_target_column(question, schema, df) or schema.get("sales_column")

    if not cust_col or df is None:
        return {"answer": "Customer column could not be detected in your dataset.", "intent": "customer", "confidence": 0.4}

    n = _extract_n(question, default=5)
    agg = df.groupby(cust_col)[val_col].sum().sort_values(ascending=False)
    top = agg.head(n)

    lines = "\n".join(f"  {i+1}. **{c}** — {_fmt(v)}" for i, (c, v) in enumerate(top.items()))
    total_customers = df[cust_col].nunique()

    answer = f"**Top {n} customers by {val_col}:**\n{lines}\n\n📊 Total unique customers: **{total_customers:,}**"
    return {"answer": answer, "intent": "customer", "confidence": 0.9}


def _answer_profit(question: str, artifacts: DatasetArtifacts) -> dict:
    schema  = artifacts.schema
    kpi     = artifacts.kpi
    metrics = artifacts.metrics
    df      = artifacts.df

    profit_col = schema.get("profit_column")
    sales_col  = schema.get("sales_column")

    parts = []

    # From metrics
    if metrics:
        margin = metrics.get("profit_margin")
        if margin is not None:
            parts.append(f"**Profit margin:** {round(float(margin) * 100, 2)}%")
        total_profit = metrics.get("total_profit")
        if total_profit is not None:
            parts.append(f"**Total profit:** {_fmt(total_profit)}")

    # From KPI
    if profit_col and kpi and profit_col in kpi:
        val = kpi[profit_col].get("sum")
        if val is not None and "Total profit" not in "\n".join(parts):
            parts.append(f"**Total {profit_col}:** {_fmt(val)}")

    # Compute from df
    if df is not None and profit_col and profit_col in df.columns:
        if not parts:
            total = df[profit_col].sum()
            avg   = df[profit_col].mean()
            max_p = df[profit_col].max()
            parts.append(f"**Total {profit_col}:** {_fmt(total)}")
            parts.append(f"**Average {profit_col}:** {_fmt(avg)}")
            parts.append(f"**Best single record:** {_fmt(max_p)}")
    elif df is not None and sales_col and sales_col in df.columns:
        # Estimate if no profit col
        cost_col = schema.get("cost_column") or schema.get("price_column")
        if cost_col and cost_col in df.columns:
            est_profit = (df[sales_col] - df[cost_col]).sum()
            parts.append(f"**Estimated total profit** (sales − cost): {_fmt(est_profit)}")

    if not parts:
        return {"answer": "Profit data could not be found. The dataset may not have a profit or cost column.", "intent": "profit", "confidence": 0.4}

    return {"answer": "\n\n".join(parts), "intent": "profit", "confidence": 0.9}


def _answer_loss(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df
    metrics = artifacts.metrics

    profit_col = schema.get("profit_column")

    if metrics:
        margin = metrics.get("profit_margin")
        if margin is not None and float(margin) < 0:
            return {"answer": f"⚠️ The dataset shows a **net loss** with a profit margin of **{round(float(margin)*100,2)}%**.", "intent": "loss", "confidence": 0.95}
        if margin is not None and float(margin) >= 0:
            return {"answer": f"✅ The dataset is **profitable** with a margin of **{round(float(margin)*100,2)}%**. No net loss detected.", "intent": "loss", "confidence": 0.95}

    if df is not None and profit_col and profit_col in df.columns:
        loss_rows = df[df[profit_col] < 0]
        total_loss = loss_rows[profit_col].sum()
        count_loss = len(loss_rows)
        return {
            "answer": (
                f"**Loss analysis:**\n"
                f"  • Loss-making records: **{count_loss:,}**\n"
                f"  • Total cumulative loss: **{_fmt(total_loss)}**"
            ),
            "intent": "loss",
            "confidence": 0.9
        }

    return {"answer": "Loss data could not be computed. No profit column was detected.", "intent": "loss", "confidence": 0.4}


def _answer_trend(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df

    date_col  = schema.get("date_column")
    sales_col = _find_target_column(question, schema, df) or schema.get("sales_column")

    if not date_col or df is None:
        return {"answer": "A date column is required for trend analysis, but none was detected.", "intent": "trend", "confidence": 0.5}

    if date_col not in df.columns:
        return {"answer": f"The detected date column '{date_col}' is not present in the data.", "intent": "trend", "confidence": 0.4}

    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])

        if re.search(r"\byear\b|\bannual\b", question.lower()):
            freq, label = "Y", "yearly"
        elif re.search(r"\bquarter\b|\bq[1-4]\b", question.lower()):
            freq, label = "Q", "quarterly"
        else:
            freq, label = "M", "monthly"

        monthly = ts.set_index(date_col).resample(freq)[sales_col].sum()

        if monthly.empty:
            return {"answer": "Not enough date data to compute trends.", "intent": "trend", "confidence": 0.5}

        lines = "\n".join(f"  • **{str(idx)[:7]}**: {_fmt(val)}" for idx, val in monthly.tail(12).items())

        # Growth rate
        if len(monthly) >= 2:
            first = monthly.iloc[0]
            last  = monthly.iloc[-1]
            if first != 0:
                growth = ((last - first) / abs(first)) * 100
                growth_str = f"\n\n📈 **Overall growth:** {growth:+.1f}% from {str(monthly.index[0])[:7]} to {str(monthly.index[-1])[:7]}"
            else:
                growth_str = ""
        else:
            growth_str = ""

        return {
            "answer": f"**{label.capitalize()} {sales_col} trend:**\n{lines}{growth_str}",
            "intent": "trend",
            "confidence": 0.9
        }

    except Exception as e:
        logger.warning(f"Trend calculation failed: {e}")
        return {"answer": "I encountered an error computing the trend. Please verify the date column format.", "intent": "trend", "confidence": 0.4}


def _answer_forecast(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df     = artifacts.df

    date_col  = schema.get("date_column")
    sales_col = schema.get("sales_column")

    if not date_col or not sales_col or df is None:
        return {"answer": "Forecasting requires both a date column and a sales/revenue column, which could not be detected.", "intent": "forecast", "confidence": 0.4}

    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col, sales_col])
        monthly = ts.set_index(date_col).resample("M")[sales_col].sum()

        if len(monthly) < 2:
            return {"answer": "Not enough historical data points to make a forecast.", "intent": "forecast", "confidence": 0.5}

        # Simple moving average forecast
        growth_rate = monthly.pct_change().mean()
        last_val    = monthly.iloc[-1]
        next_val    = last_val * (1 + growth_rate)

        last_month  = monthly.index[-1]
        # Next month
        next_month  = last_month + pd.DateOffset(months=1)

        direction = "📈 increase" if growth_rate > 0 else "📉 decrease"

        answer = (
            f"**Forecast for {str(next_month)[:7]}:**\n\n"
            f"  • Predicted {sales_col}: **{_fmt(next_val)}**\n"
            f"  • Based on average monthly growth rate of **{growth_rate*100:+.1f}%**\n"
            f"  • Last recorded ({str(last_month)[:7]}): {_fmt(last_val)}\n\n"
            f"Trend suggests a {direction} in the next period."
        )
        return {"answer": answer, "intent": "forecast", "confidence": 0.8}

    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
        return {"answer": "I encountered an error while computing the forecast.", "intent": "forecast", "confidence": 0.4}


def _answer_anomaly(artifacts: DatasetArtifacts) -> dict:
    insights_data = artifacts.insights

    if isinstance(insights_data, dict):
        all_insights = insights_data.get("insights", [])
    elif isinstance(insights_data, list):
        all_insights = insights_data
    else:
        all_insights = []

    anomaly_insights = [i for i in all_insights if "anomal" in i.get("type", "").lower() or "outlier" in i.get("description", "").lower()]

    if anomaly_insights:
        lines = "\n".join(f"  ⚠️ {i['description']}" for i in anomaly_insights)
        return {"answer": f"**Anomalies detected:**\n{lines}", "intent": "anomaly", "confidence": 0.95}

    # Compute from dataset
    schema = artifacts.schema
    df     = artifacts.df
    sales_col = schema.get("sales_column")

    if df is not None and sales_col and sales_col in df.columns:
        try:
            from scipy import stats as sp_stats
            col_data = df[sales_col].dropna()
            z_scores = np.abs(sp_stats.zscore(col_data))
            outlier_count = int((z_scores > 3).sum())
            if outlier_count > 0:
                return {
                    "answer": f"⚠️ **{outlier_count} statistical anomalies** detected in `{sales_col}` (values beyond 3 standard deviations from the mean). These could represent data entry errors or exceptional events.",
                    "intent": "anomaly",
                    "confidence": 0.85
                }
            else:
                return {"answer": f"✅ **No significant anomalies** detected in `{sales_col}`. All values are within 3 standard deviations of the mean.", "intent": "anomaly", "confidence": 0.85}
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

    if not all_insights:
        return {"answer": "No anomaly analysis is available yet. Please ensure the dataset pipeline has completed successfully.", "intent": "anomaly", "confidence": 0.4}

    return {"answer": "✅ No anomalies were detected by the AI analysis engine.", "intent": "anomaly", "confidence": 0.8}


def _answer_insights(artifacts: DatasetArtifacts) -> dict:
    insights_data = artifacts.insights

    if isinstance(insights_data, dict):
        all_insights = insights_data.get("insights", [])
        summary      = insights_data.get("summary", "")
    elif isinstance(insights_data, list):
        all_insights = insights_data
        summary      = ""
    else:
        all_insights = []
        summary      = ""

    if not all_insights and not summary:
        return {"answer": "AI insights are not yet available. Please ensure the dataset pipeline completed successfully.", "intent": "insight", "confidence": 0.5}

    parts = []
    if summary:
        parts.append(f"**Summary:** {summary}")

    icon_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    for i in all_insights[:8]:
        sev  = i.get("severity", "info")
        icon = icon_map.get(sev, "ℹ️")
        parts.append(f"{icon} {i.get('description', '')}")

    return {"answer": "\n\n".join(parts), "intent": "insight", "confidence": 0.95}


def _answer_comparison(question: str, artifacts: DatasetArtifacts) -> dict:
    """Try to compare two named entities from the question."""
    schema = artifacts.schema
    df     = artifacts.df

    # Look for two quoted or capitalized terms
    matches = re.findall(r'"([^"]+)"', question)
    if len(matches) < 2:
        # Try unquoted capitalized words
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)

    if len(matches) >= 2 and df is not None:
        a, b = matches[0], matches[1]
        # Find which column contains these values
        for col in df.select_dtypes(include=["object", "category"]).columns:
            col_vals = df[col].astype(str).str.lower()
            if a.lower() in col_vals.values and b.lower() in col_vals.values:
                val_col = schema.get("sales_column")
                if val_col and val_col in df.columns:
                    a_val = df[df[col].str.lower() == a.lower()][val_col].sum()
                    b_val = df[df[col].str.lower() == b.lower()][val_col].sum()
                    winner = a if a_val > b_val else b
                    return {
                        "answer": (
                            f"**Comparison of {a} vs {b} by {val_col}:**\n\n"
                            f"  • **{a}**: {_fmt(a_val)}\n"
                            f"  • **{b}**: {_fmt(b_val)}\n\n"
                            f"🏆 **{winner}** is higher."
                        ),
                        "intent": "comparison",
                        "confidence": 0.85
                    }

    return {"answer": "Please specify two items to compare, e.g., *\"Compare 'North' vs 'South' region\"*.", "intent": "comparison", "confidence": 0.5}


# ─── Helper: Find Target Column ───────────────────────────────────────────────

def _find_target_column(question: str, schema: dict, df: "pd.DataFrame | None") -> str | None:
    """Try to match a column name mentioned in the question."""
    q = question.lower()

    # Check schema roles
    role_keywords = {
        "sales": ["sales", "revenue", "income"],
        "profit": ["profit", "margin", "earning"],
        "cost": ["cost", "expense", "spend"],
        "quantity": ["quantity", "qty", "units", "volume"],
        "price": ["price", "rate", "per unit"],
        "region": ["region", "country", "location"],
        "product": ["product", "item", "sku"],
        "customer": ["customer", "client"],
        "date": ["date", "time", "month"],
    }
    for role, keywords in role_keywords.items():
        if any(kw in q for kw in keywords):
            schema_key = f"{role}_column"
            if schema.get(schema_key):
                return schema.get(schema_key)

    # Try exact column name match
    if df is not None:
        for col in df.columns:
            if col.lower() in q or col.lower().replace("_", " ") in q:
                return col

    return None


# ─── Main Dispatcher ──────────────────────────────────────────────────────────

def answer_question(question: str, artifacts: DatasetArtifacts) -> dict:
    """Route question to the right handler and return answer dict."""
    if not artifacts.available:
        return {
            "answer": (
                "⏳ The dataset is still being processed or the artifacts are unavailable. "
                "Please wait a moment and try again."
            ),
            "intent": "unavailable",
            "confidence": 0.0
        }

    intent = detect_intent(question)

    try:
        if intent == "greeting":
            return _answer_greeting(question)
        elif intent == "summary":
            return _answer_summary(artifacts)
        elif intent == "columns":
            return _answer_columns(artifacts)
        elif intent == "rows":
            return _answer_rows(artifacts)
        elif intent == "missing":
            return _answer_missing(artifacts)
        elif intent in ("total_sales", "total_generic"):
            return _answer_total(question, artifacts)
        elif intent == "average":
            return _answer_average(question, artifacts)
        elif intent == "maximum":
            return _answer_max(question, artifacts)
        elif intent == "minimum":
            return _answer_min(question, artifacts)
        elif intent == "count":
            return _answer_count(question, artifacts)
        elif intent == "top_n":
            return _answer_top_n(question, artifacts)
        elif intent == "region":
            return _answer_region(question, artifacts)
        elif intent == "product":
            return _answer_product(question, artifacts)
        elif intent == "customer":
            return _answer_customer(question, artifacts)
        elif intent == "profit":
            return _answer_profit(question, artifacts)
        elif intent == "loss":
            return _answer_loss(question, artifacts)
        elif intent == "trend":
            return _answer_trend(question, artifacts)
        elif intent == "forecast":
            return _answer_forecast(question, artifacts)
        elif intent == "anomaly":
            return _answer_anomaly(artifacts)
        elif intent == "insight":
            return _answer_insights(artifacts)
        elif intent == "comparison":
            return _answer_comparison(question, artifacts)
        else:
            # Unknown: try a broad keyword search across KPI data
            kpi = artifacts.kpi
            if kpi:
                schema = artifacts.schema
                target = _find_target_column(question, schema, artifacts.df)
                if target and target in kpi:
                    k = kpi[target]
                    lines = "\n".join(f"  • **{stat}**: {_fmt(val)}" for stat, val in k.items() if stat not in ("column",))
                    return {"answer": f"**Statistics for {target}:**\n{lines}", "intent": "stats", "confidence": 0.7}

            return {
                "answer": (
                    "I'm not sure how to answer that. Here are some things you can ask:\n\n"
                    "• *\"What is the total sales?\"*\n"
                    "• *\"Show me the top 5 products\"*\n"
                    "• *\"What are the monthly trends?\"*\n"
                    "• *\"Give me an overview\"*\n"
                    "• *\"Are there any anomalies?\"*"
                ),
                "intent": "unknown",
                "confidence": 0.0
            }
    except Exception as e:
        logger.error(f"Answer generation failed for intent '{intent}': {e}", exc_info=True)
        return {
            "answer": "I encountered an internal error while processing your question. Please try rephrasing it.",
            "intent": intent,
            "confidence": 0.0
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset Q&A Query Engine")
    parser.add_argument("--user_id",    default="default_user")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--question",   required=True)
    parser.add_argument("--dataset_dir", default=None,
                        help="Override dataset artifact directory (optional)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or resolve_dataset_dir(args.user_id, args.dataset_id)

    artifacts = DatasetArtifacts(dataset_dir)
    result    = answer_question(args.question, artifacts)

    # Single JSON line to stdout — Node.js parses this
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()