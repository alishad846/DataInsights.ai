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
    return os.path.normpath(os.path.join(BASE_DATA_DIR, user_id, dataset_id))


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

    def __init__(self, dataset_dir: str, csv_file: str = None):
        self.dir = dataset_dir
        self.csv_file = csv_file  # Explicit path to the CSV; overrides dir-based lookup
        self._schema = None
        self._kpi = None
        self._metrics = None
        self._insights = None
        self._feature_importance = None
        self._profile = None
        self._df = None

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
    def feature_importance(self) -> dict:
        if self._feature_importance is None:
            self._feature_importance = (
                _load_json(os.path.join(self.dir, "feature_importance.json")) or {}
            )
        return self._feature_importance

    @property
    def profile(self) -> dict:
        if self._profile is None:
            self._profile = (
                _load_json(os.path.join(self.dir, "profile_report.json")) or {}
            )
        return self._profile

    @property
    def df(self) -> "pd.DataFrame | None":
        if self._df is None:
            # Priority 1: explicit csv_file path passed from Node.js
            if self.csv_file and os.path.exists(self.csv_file):
                self._df = _load_csv(self.csv_file)
                logger.info(f"[DatasetArtifacts] Loaded CSV from explicit path: {self.csv_file}")
            else:
                # Priority 2: look for cleaned_data.csv in the artifact dir
                cleaned = os.path.join(self.dir, "cleaned_data.csv")
                self._df = _load_csv(cleaned)
                if self._df is not None:
                    logger.info(f"[DatasetArtifacts] Loaded CSV from artifact dir: {cleaned}")
                else:
                    logger.warning(f"[DatasetArtifacts] No CSV found. csv_file={self.csv_file}, dir={self.dir}")
        return self._df

    @property
    def available(self) -> bool:
        return bool(self.schema) or (self.df is not None)


# ─── Intent Router ───────────────────────────────────────────────────────────

INTENTS = {
    "greeting": [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bgreet",
        r"\bgood\s+(morning|afternoon|evening)\b",
        r"\bwhat can you (do|help|answer)\b",
        r"\bhelp\b",
        r"\bwhat are you\b",
        r"\babilities\b",
    ],
    "total_sales": [
        r"\btotal\s+sales?\b",
        r"\btotal\s+revenue\b",
        r"\bsum\s+of\s+sales?\b",
        r"\boverall\s+sales?\b",
        r"\bgross\s+sales?\b",
        r"\bhow\s+much\s+(did|were)\s+(we\s+)?sell\b",
        r"\bsales?\s+total\b",
    ],
    "total_generic": [
        r"\btotal\b",
        r"\bsum\s+of\b",
        r"\baggregate\b",
    ],
    "average": [
        r"\baverage\b",
        r"\bavg\b",
        r"\bmean\b",
        r"\bper\s+\w+\b",
        r"\btypical\b",
    ],
    "maximum": [
        r"\bmax(imum)?\b",
        r"\bhighest\b",
        r"\bbiggest\b",
        r"\bmost\b",
        r"\btop\b",
        r"\bbest\b",
        r"\bpeak\b",
        r"\bgreatest\b",
    ],
    "minimum": [
        r"\bmin(imum)?\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bworst\b",
        r"\bbottom\b",
        r"\bweakest\b",
    ],
    "top_n": [
        r"\btop\s+\d+\b",
        r"\bbest\s+\d+\b",
        r"\bhighest\s+\d+\b",
    ],
    "count": [
        r"\bhow\s+many\b",
        r"\bcount\b",
        r"\bnumber\s+of\b",
        r"\bquantity\b",
        r"\btotal\s+records?\b",
        r"\btotal\s+rows?\b",
        r"\btotal\s+entries\b",
    ],
    "region": [
        r"\bregion\b",
        r"\bcountry\b",
        r"\bcountries\b",
        r"\bcity\b",
        r"\bcities\b",
        r"\blocation\b",
        r"\bterritory\b",
        r"\bzone\b",
        r"\barea\b",
        r"\bmarket\b",
        r"\bby\s+region\b",
        r"\bby\s+location\b",
        r"\bby\s+country\b",
        r"\bstate\b",
    ],
    "product": [
        r"\bproduct\b",
        r"\bproducts\b",
        r"\bitem\b",
        r"\bitems\b",
        r"\bsku\b",
        r"\bbest.sell\w+\b",
        r"\btop.sell\w+\b",
        r"\bgoods\b",
    ],
    "customer": [
        r"\bcustomer\b",
        r"\bcustomers\b",
        r"\bclient\b",
        r"\bclients\b",
        r"\bbuyer\b",
        r"\bbuyers\b",
        r"\baccount\b",
    ],
    "profit": [
        r"\bprofit\b",
        r"\bprofitable\b",
        r"\bprofitability\b",
        r"\bmargin\b",
        r"\bnet\s+income\b",
        r"\bearnings\b",
    ],
    "loss": [
        r"\bloss\b",
        r"\blosses\b",
        r"\bnegative\b",
        r"\bdeficit\b",
        r"\bin\s+the\s+red\b",
    ],
    "trend": [
        r"\btrend\b",
        r"\bover\s+time\b",
        r"\bmonthly\b",
        r"\bby\s+month\b",
        r"\bquarterly\b",
        r"\bby\s+year\b",
        r"\bannual\b",
        r"\bhistorical\b",
        r"\btime\s+series\b",
        r"\bgrowth\b",
    ],
    "forecast": [
        r"\bforecast\b",
        r"\bpredict\b",
        r"\bnext\s+(month|quarter|year)\b",
        r"\bfuture\b",
        r"\bprojection\b",
        r"\bexpect\b",
        r"\bestimate\b",
    ],
    "anomaly": [
        r"\banomal\w+\b",
        r"\boutlier\b",
        r"\bspike\b",
        r"\bunusual\b",
        r"\babnormal\b",
        r"\bweird\b",
        r"\bstrange\b",
    ],
    "comparison": [
        r"\bcompar\w+\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+is\s+(better|worse|higher|lower)\b",
    ],
    "sales_strategy": [
        r"\brecommend\w*\b",
        r"\bsuggest\w*\b",
        r"\bwhat\s+can\s+i\s+do\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bhow\s+can\s+i\b",
        r"\bhow\s+do\s+i\b",
        r"\bimprove\s+sales\b",
        r"\bincrease\s+sales\b",
        r"\bboost\s+sales\b",
        r"\bimprove\s+revenue\b",
        r"\bincrease\s+revenue\b",
        r"\bboost\s+revenue\b",
        r"\baction\s+plan\b",
        r"\bnext\s+steps\b",
        r"\bfocus\s+on\b",
        r"\bwhat\s+to\s+do\b",
        r"\bfactor\w*\b",
        r"\baffect\w*\b",
        r"\bimpact\w*\b",
        r"\bperformer\b",
        r"\bbest\s+performer\b",
        r"\bworst\s+performer\b",
    ],
    "recommendation": [
        r"\brecommend\w*\b",
        r"\bsuggest\w*\b",
        r"\bwhat\s+can\s+i\s+do\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bhow\s+can\s+i\b",
        r"\bhow\s+do\s+i\b",
        r"\bimprove\s+sales\b",
        r"\bincrease\s+sales\b",
        r"\bboost\s+sales\b",
        r"\bimprove\s+revenue\b",
        r"\bincrease\s+revenue\b",
        r"\bboost\s+revenue\b",
        r"\baction\s+plan\b",
        r"\bnext\s+steps\b",
        r"\bfocus\s+on\b",
        r"\bwhat\s+to\s+do\b",
    ],
    "driver": [
        r"\bdriv\w*\b",
        r"\bcontribut\w*\b",
        r"\bwhat\s+drives\b",
        r"\bwhat\s+is\s+driving\b",
        r"\bwhich\s+.*\bdriving\b",
        r"\bwhat\s+is\s+behind\b",
    ],
    "insight": [
        r"\binsight\b",
        r"\banalysis\b",
        r"\banalyze\b",
    ],
    "summary": [
        r"\bsummary\b",
        r"\boverview\b",
        r"\bsummariz\w+\b",
        r"\bdescrib\w+\b",
        r"\btell\s+me\s+about\b",
        r"\bwhat\s+is\s+in\b",
        r"\bwhat\s+does\s+the\s+data\b",
        r"\babout\s+the\s+data\b",
    ],
    "advanced_mode": [
        r"\badvanced\s+chat\b",
        r"\badvanced\s+analysis\b",
        r"\badvanced\s+dataset\b",
        r"\badvanced\s+data\s+set\b",
        r"\bfull\s+dataset\b",
        r"\bcomplete\s+dataset\b",
        r"\bfull\s+data\b",
        r"\bcomplete\s+data\b",
        r"\bdeep\s+analysis\b",
        r"\bmore\s+concise\s+answer\b",
    ],
    "columns": [
        r"\bcolumn\b",
        r"\bfield\b",
        r"\bheader\b",
        r"\bvariable\b",
        r"\bwhat\s+data\b",
        r"\bwhat\s+columns\b",
        r"\bwhat\s+fields\b",
        r"\bstructure\b",
    ],
    "rows": [
        r"\brow\b",
        r"\brecord\b",
        r"\bentry\b",
        r"\bdata\s+point\b",
        r"\bsize\s+of\b",
        r"\bhow\s+large\b",
    ],
    "missing": [
        r"\bmissing\b",
        r"\bnull\b",
        r"\bnan\b",
        r"\bempty\b",
        r"\bblank\b",
        r"\bincomplete\b",
    ],
}


def detect_intent(question: str) -> str:
    """Returns the best matching intent for a question."""
    q = question.lower()

    # Priority intents (checked first)
    priority_order = [
        "greeting",
        "top_n",
        "total_sales",
        "driver",
        "anomaly",
        "forecast",
        "trend",
        "region",
        "product",
        "customer",
        "profit",
        "loss",
        "columns",
        "rows",
        "missing",
        "summary",
        "advanced_mode",
        "sales_strategy",
        "recommendation",
        "insight",
        "count",
        "maximum",
        "minimum",
        "average",
        "total_generic",
        "comparison",
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
    m = re.search(
        r"\b(?:top|best|highest|bottom|lowest|worst)\s+(\d+)\b", question.lower()
    )
    if m:
        return int(m.group(1))
    return default


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _column_name_contains(col: str, keywords: list[str]) -> bool:
    col_name = _normalize_name(col)
    return any(_normalize_name(keyword) in col_name for keyword in keywords)


def _is_id_like_column(col: str) -> bool:
    return bool(
        re.search(r"\b(id|uuid|code|zip|postal|reference|ref|key)\b", _normalize_name(col))
    )


def _is_numeric_column(df: "pd.DataFrame | None", col: str) -> bool:
    if df is None or col not in df.columns:
        return False

    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        return True

    try:
        coerced = pd.to_numeric(series, errors="coerce")
    except Exception:
        return False

    valid = int(coerced.notna().sum())
    if valid == 0:
        return False

    return valid >= max(3, int(len(series) * 0.8)) and coerced.nunique(dropna=True) > 1


def _is_measure_like_column(col: str) -> bool:
    name = _normalize_name(col)
    return any(
        term in name
        for term in [
            "sales",
            "revenue",
            "profit",
            "margin",
            "quantity",
            "qty",
            "unit price",
            "price",
            "discount",
            "cost",
            "amount",
            "value",
            "net",
            "gross",
        ]
    )


def _humanize_group_label(col: str) -> str:
    name = _normalize_name(col)
    if "sub category" in name or "subcategory" in name:
        return "sub-categories"
    if "category" in name:
        return "categories"
    if "product" in name or "item" in name or "sku" in name:
        return "products"
    if "region" in name:
        return "regions"
    if "country" in name:
        return "countries"
    if "state" in name:
        return "states"
    if "city" in name:
        return "cities"
    if "customer" in name:
        return "customers"
    if "ship" in name or "shipping" in name or "delivery" in name:
        return "ship modes"
    return col


def _resolve_metric_column(
    question: str, schema: dict, df: "pd.DataFrame | None"
) -> str | None:
    if df is None:
        return None

    q = question.lower()
    role_terms = {
        "sales": ["sales", "revenue", "gross sales", "net sales", "turnover"],
        "profit": ["profit", "margin", "earnings", "income"],
        "quantity": ["quantity", "qty", "units", "volume"],
        "discount": ["discount", "rebate", "markdown"],
        "price": ["price", "unit price", "per unit", "rate"],
        "cost": ["cost", "expense", "spend"],
    }

    role_order = []
    if any(term in q for term in ["profit", "margin", "earning", "income"]):
        role_order.append("profit")
    if any(term in q for term in ["quantity", "qty", "units", "volume", "count"]):
        role_order.append("quantity")
    if any(term in q for term in ["discount", "rebate", "markdown"]):
        role_order.append("discount")
    if any(term in q for term in ["price", "unit price", "per unit", "rate"]):
        role_order.append("price")
    if any(term in q for term in ["cost", "expense", "spend"]):
        role_order.append("cost")
    if any(
        term in q
        for term in [
            "sales",
            "revenue",
            "sold",
            "selling",
            "drive",
            "driving",
            "top",
            "most",
            "highest",
            "best",
        ]
    ):
        role_order.append("sales")
    role_order.extend(["sales", "profit", "quantity", "discount", "price", "cost"])

    seen = set()
    for role in role_order:
        if role in seen:
            continue
        seen.add(role)

        schema_key = f"{role}_column"
        candidates = []

        schema_col = schema.get(schema_key)
        if schema_col:
            candidates.append(schema_col)

        candidates.extend(
            [c for c in df.columns if _column_name_contains(c, role_terms[role])]
        )

        for candidate in candidates:
            if candidate in df.columns and _is_numeric_column(df, candidate):
                return candidate

    numeric_candidates = [c for c in df.columns if _is_numeric_column(df, c)]
    measure_candidates = [
        c
        for c in numeric_candidates
        if _is_measure_like_column(c) and not _is_id_like_column(c)
    ]
    if measure_candidates:
        return measure_candidates[0]

    non_id_numeric = [c for c in numeric_candidates if not _is_id_like_column(c)]
    if non_id_numeric:
        return non_id_numeric[0]

    return numeric_candidates[0] if numeric_candidates else None


def _resolve_group_column(
    question: str,
    schema: dict,
    df: "pd.DataFrame | None",
    metric_col: str | None = None,
) -> tuple[str | None, str]:
    if df is None:
        return None, "category"

    q = question.lower()
    role_keywords = {
        "product": ["product", "category", "sub category", "subcategory", "item", "sku", "brand"],
        "region": ["region", "country", "state", "city", "territory", "zone", "market", "location", "area"],
        "customer": ["customer", "client", "buyer", "account"],
        "shipping": ["ship", "shipping", "delivery", "mode"],
        "time": ["date", "time", "month", "quarter", "year", "week"],
    }

    role_order = []
    if any(term in q for term in ["product", "category", "item", "sku", "brand"]):
        role_order.append("product")
    if any(term in q for term in ["region", "country", "state", "city", "location", "market", "territory", "zone"]):
        role_order.append("region")
    if any(term in q for term in ["customer", "client", "buyer", "account"]):
        role_order.append("customer")
    if any(term in q for term in ["ship", "shipping", "delivery", "mode"]):
        role_order.append("shipping")
    if any(term in q for term in ["date", "month", "quarter", "year", "week", "time"]):
        role_order.append("time")
    role_order.extend(["product", "region", "customer", "shipping", "time"])

    seen_roles = set()
    best_col = None
    best_score = float("-inf")

    for role in role_order:
        if role in seen_roles:
            continue
        seen_roles.add(role)

        schema_key = {
            "product": "product_column",
            "region": "region_column",
            "customer": "customer_column",
            "shipping": "shipping_column",
            "time": "date_column",
        }.get(role)

        candidates = []
        if schema_key and schema.get(schema_key):
            candidates.append(schema.get(schema_key))

        candidates.extend([c for c in df.columns if _column_name_contains(c, role_keywords[role])])

        for candidate in candidates:
            if candidate not in df.columns:
                continue

            score = 0
            col_name = _normalize_name(candidate)
            if metric_col and candidate == metric_col:
                score -= 100

            if any(term in col_name for term in role_keywords[role]):
                score += 80

            if role == "product":
                if any(term in q for term in ["category", "subcategory", "sub-category"]):
                    if "category" in col_name or "subcategory" in col_name:
                        score += 35
                if any(term in q for term in ["product", "item", "sku", "brand"]):
                    if any(term in col_name for term in ["product", "item", "sku", "brand"]):
                        score += 35

            if role == "region" and any(
                term in q for term in ["region", "country", "state", "city", "location", "market", "territory"]
            ):
                score += 35

            if role == "customer" and any(term in q for term in ["customer", "client", "buyer", "account"]):
                score += 35

            if role == "shipping" and any(term in q for term in ["ship", "shipping", "delivery", "mode"]):
                score += 25

            if _is_id_like_column(candidate):
                score -= 60
            if _is_numeric_column(df, candidate):
                score -= 20

            if score > best_score:
                best_col = candidate
                best_score = score

    if best_col:
        return best_col, _humanize_group_label(best_col)

    categorical_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c != metric_col and not _is_id_like_column(c)
    ]
    if categorical_cols:
        preferred = [
            c
            for c in categorical_cols
            if any(
                kw in _normalize_name(c)
                for kw in [
                    "product",
                    "category",
                    "subcategory",
                    "sub category",
                    "region",
                    "country",
                    "state",
                    "city",
                    "customer",
                    "ship",
                    "shipping",
                    "delivery",
                ]
            )
        ]
        fallback = preferred or categorical_cols
        return fallback[0], _humanize_group_label(fallback[0])

    return None, "category"


def _coerce_numeric_series(df: "pd.DataFrame", col: str) -> "pd.Series | None":
    if df is None or col not in df.columns:
        return None
    try:
        return pd.to_numeric(df[col], errors="coerce")
    except Exception:
        return None


def _aggregate_sales_by_dimension(
    df: "pd.DataFrame",
    sales_col: str,
    group_col: str,
    value_name: str = "_sales_metric",
) -> "pd.Series | None":
    if df is None or sales_col not in df.columns or group_col not in df.columns:
        return None

    sales_series = _coerce_numeric_series(df, sales_col)
    if sales_series is None:
        return None

    working = df[[group_col]].copy()
    working[value_name] = sales_series
    working = working.dropna(subset=[group_col, value_name])
    if working.empty:
        return None

    agg = working.groupby(group_col)[value_name].sum().sort_values(ascending=False)
    return agg if len(agg) >= 2 else None


def _format_ranked_lines(
    agg: "pd.Series", limit: int = 5, indent: str = "  "
) -> str:
    lines = []
    for i, (name, value) in enumerate(agg.head(limit).items()):
        lines.append(f"{indent}{i + 1}. **{name}** - {_fmt(value)}")
    return "\n".join(lines)


def _display_column_name(col: str | None) -> str:
    if not col:
        return "this dataset"
    return str(col).replace("_", " ").strip()


def _friendly_group_label(col: str | None) -> str:
    if not col:
        return "segment"
    normalized = _normalize_name(col)
    if "customer" in normalized:
        return "customers"
    if any(term in normalized for term in ["salesperson", "sales rep", "sales rep", "rep"]):
        return "sales reps"
    if any(term in normalized for term in ["product", "item", "sku", "brand"]):
        return "products"
    if "region" in normalized:
        return "regions"
    if "country" in normalized:
        return "countries"
    if "state" in normalized:
        return "states"
    if "city" in normalized:
        return "cities"
    if any(term in normalized for term in ["channel", "campaign", "source"]):
        return "channels"
    if "warehouse" in normalized or "depot" in normalized:
        return "warehouses"
    if "status" in normalized:
        return "statuses"
    return _display_column_name(col)


def _singular_group_label(col: str | None) -> str:
    label = _friendly_group_label(col)
    singular_map = {
        "customers": "customer",
        "sales reps": "sales rep",
        "products": "product",
        "regions": "region",
        "countries": "country",
        "states": "state",
        "cities": "city",
        "channels": "channel",
        "warehouses": "warehouse",
        "statuses": "status",
    }
    return singular_map.get(label, label[:-1] if label.endswith("s") else label)


def _infer_sales_performer(
    df: "pd.DataFrame",
    sales_col: str,
    candidates: list[tuple[str, str]],
) -> dict | None:
    best = None
    best_score = float("-inf")

    for label, col in candidates:
        agg = _aggregate_sales_by_dimension(df, sales_col, col)
        if agg is None or agg.empty:
            continue

        spread = float(agg.iloc[0] - agg.iloc[-1])
        ratio = float(agg.iloc[0] / max(abs(agg.iloc[-1]), 1.0))
        score = spread * (1.0 + np.log1p(len(agg))) + ratio

        if score > best_score:
            best_score = score
            best = {
                "label": label,
                "column": col,
                "best_name": agg.idxmax(),
                "best_value": float(agg.max()),
                "worst_name": agg.idxmin(),
                "worst_value": float(agg.min()),
                "top5": agg.head(5),
            }

    return best


def _infer_sales_factors(
    df: "pd.DataFrame",
    sales_col: str,
    profit_col: str | None = None,
    feature_importance: dict | None = None,
) -> dict:
    sales_series = _coerce_numeric_series(df, sales_col)
    if sales_series is None:
        return {"numeric": [], "categorical": [], "model": []}

    numeric_factors = []
    for col in df.select_dtypes(include=["number"]).columns:
        if col == sales_col or _is_id_like_column(col):
            continue

        other = _coerce_numeric_series(df, col)
        if other is None:
            continue

        paired = pd.concat([sales_series, other], axis=1).dropna()
        if len(paired) < 8 or paired.iloc[:, 1].nunique(dropna=True) < 2:
            continue

        corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
        if pd.notna(corr):
            numeric_factors.append((col, float(corr), abs(float(corr))))

    numeric_factors.sort(key=lambda item: item[2], reverse=True)

    categorical_factors = []
    candidate_groups = [
        ("product", ["product", "category", "item", "brand", "sku"]),
        ("region", ["region", "country", "state", "city", "territory", "market"]),
        ("channel", ["channel", "campaign", "source", "warehouse", "salesperson", "customer"]),
    ]

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == sales_col or _is_id_like_column(col):
            continue

        normalized = _normalize_name(col)
        if not any(k in normalized for _, keys in candidate_groups for k in keys):
            continue

        agg = _aggregate_sales_by_dimension(df, sales_col, col)
        if agg is None or agg.empty:
            continue

        spread = float(agg.iloc[0] - agg.iloc[-1])
        score = spread * (1.0 + np.log1p(len(agg)))
        categorical_factors.append((col, float(agg.iloc[0]), float(agg.iloc[-1]), score))

    categorical_factors.sort(key=lambda item: item[3], reverse=True)

    model_factors = []
    if isinstance(feature_importance, dict):
        importance_map = feature_importance.get("importance")
        if isinstance(importance_map, dict):
            for name, weight in list(importance_map.items())[:8]:
                if name in df.columns and name != sales_col and not _is_id_like_column(name):
                    model_factors.append((name, float(weight)))

    return {
        "numeric": numeric_factors[:5],
        "categorical": categorical_factors[:5],
        "model": model_factors[:5],
    }


def _question_has(question: str, *phrases: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in phrases)


def _group_concentration_text(
    agg: "pd.Series", label: str, sales_col: str, min_share: float = 0.35
) -> str | None:
    if agg is None or agg.empty:
        return None

    total = float(agg.sum())
    if total <= 0:
        return None

    top_n = min(3, len(agg))
    top_share = float(agg.head(top_n).sum()) / total
    if top_share < min_share:
        return None

    return (
        f"Sales are concentrated in the top {top_n} {label}s, which together account for {top_share * 100:.1f}% of {sales_col}. "
        f"That means the fastest growth lever is to scale the winning {label}s and lift the weaker half."
    )


def _build_sales_suggestions(
    question: str,
    sales_col: str,
    *,
    performer: dict | None = None,
    region_col: str | None = None,
    product_col: str | None = None,
    customer_col: str | None = None,
    salesperson_col: str | None = None,
    channel_col: str | None = None,
    campaign_col: str | None = None,
    warehouse_col: str | None = None,
    status_col: str | None = None,
    brand_col: str | None = None,
    profit_col: str | None = None,
    discount_col: str | None = None,
    date_col: str | None = None,
    top_numeric: list[tuple[str, float, float]] | None = None,
) -> list[str]:
    primary: list[str] = []
    secondary: list[str] = []
    q = question.lower()

    def add(text: str | None, bucket: str = "secondary"):
        if not text:
            return
        if bucket == "primary":
            if text not in primary:
                primary.append(text)
        else:
            if text not in secondary:
                secondary.append(text)

    def prioritise_performer_questions(label: str, best_name: str | None, worst_name: str | None):
        add(f"Why is {worst_name} underperforming?", "primary")
        add(f"How can I replicate the success of {best_name}?", "primary")
        add(f"Show me the top 5 {label}s by {sales_col}.", "secondary")
        add(f"Show me the bottom 5 {label}s by {sales_col}.", "secondary")

    if performer:
        label = performer.get("label") or "segment"
        best_name = performer.get("best_name")
        worst_name = performer.get("worst_name")
        prioritise_performer_questions(label, best_name, worst_name)

    if region_col:
        add(f"Which {_singular_group_label(region_col)} is underperforming?", "primary")
        add(f"Show me sales by {_friendly_group_label(region_col)}.")

    if product_col:
        add(f"Which {_singular_group_label(product_col)} should I promote more?", "primary")
        add(f"Show me the top 5 {_friendly_group_label(product_col)} by {sales_col}.")

    if customer_col:
        add(f"Which {_friendly_group_label(customer_col)} are the most valuable?", "primary")

    if salesperson_col:
        add(f"Which {_singular_group_label(salesperson_col)} is the best performer?", "primary")

    if channel_col:
        add(f"Which {_singular_group_label(channel_col)} drives the most sales?", "primary")

    if campaign_col:
        add(f"Which {_singular_group_label(campaign_col)} campaign performs best?", "primary")

    if warehouse_col:
        add(f"Which {_singular_group_label(warehouse_col)} location is weakest?", "primary")

    if status_col:
        add(f"How do sales differ by {_friendly_group_label(status_col)}?")

    if brand_col:
        add(f"Which {_singular_group_label(brand_col)} should I scale?")

    if profit_col:
        add(f"Which products or segments have the best profit margin?", "primary")
        add(f"How can I improve profit alongside sales?")

    if discount_col:
        add("Are discounts hurting sales or profit?", "primary")
        add("What discount strategy should I use?")

    if date_col:
        add(f"What are the monthly trends for {sales_col}?", "primary")
        add(f"When do {sales_col} peak and dip?", "primary")

    if top_numeric:
        for name, corr, _ in top_numeric[:2]:
            if abs(corr) >= 0.2:
                direction = "higher" if corr > 0 else "lower"
                add(f"How does {_display_column_name(name)} affect {sales_col}?", "primary")
                add(
                    f"Should I focus on {direction} {_display_column_name(name)} to improve {sales_col}?",
                    "secondary",
                )

    if _question_has(q, "increase sales", "improve sales", "boost sales", "grow sales"):
        add(f"Which product or category should I scale first?", "primary")
        add("What tactic will increase sales the fastest?", "primary")
        add("Where are discounts hurting profit?", "secondary")

    if _question_has(q, "factor", "affect", "impact", "driver"):
        add(f"Which columns are the strongest sales drivers?", "primary")
        add(f"How do {sales_col} drivers compare across segments?", "primary")

    if _question_has(q, "best performer", "worst performer", "performer"):
        add("Which product is the top seller?", "primary")
        add("Which region is the weakest?", "primary")

    if _question_has(q, "trend", "monthly", "quarterly", "yearly", "forecast"):
        add(f"What is the short-term forecast for {sales_col}?", "primary")
        add(f"Which month was the peak and which was the trough?", "primary")

    if _question_has(q, "customer", "retention", "upsell"):
        add("Which customers drive the most revenue?", "primary")
        add("How can I improve customer retention?", "primary")

    if _question_has(q, "region", "state", "city", "country"):
        add(f"Which {_singular_group_label(region_col or 'region')} should I focus on?", "primary")

    add(f"What is the total {sales_col}?", "secondary")

    suggestions = primary + [item for item in secondary if item not in primary]
    return suggestions[:5]


def _build_sales_strategy_answer(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    metrics = artifacts.metrics or {}
    df = artifacts.df
    feature_importance = artifacts.feature_importance or {}
    q = question.lower()

    if df is None:
        return {
            "answer": "I need the cleaned dataset before I can analyze sales performance. Please complete the data preparation step and try again.",
            "intent": "sales_strategy",
            "confidence": 0.4,
            "suggested_questions": [
                "What columns are available?",
                "What is the total sales?",
                "Show me the top 5 products by sales",
            ],
        }

    sales_col = _resolve_metric_column(question, schema, df) or schema.get("sales_column")
    if not sales_col or sales_col not in df.columns or not _is_numeric_column(df, sales_col):
        sales_col = schema.get("sales_column") if schema.get("sales_column") in df.columns else None

    if not sales_col or not _is_numeric_column(df, sales_col):
        return {
            "answer": "I could not find a reliable sales or revenue column in this dataset.",
            "intent": "sales_strategy",
            "confidence": 0.4,
        }

    profit_col = schema.get("profit_column") if schema.get("profit_column") in df.columns else None
    discount_col = schema.get("discount_column") if schema.get("discount_column") in df.columns else None
    date_col = schema.get("date_column") if schema.get("date_column") in df.columns else None

    region_col = schema.get("region_column") if schema.get("region_column") in df.columns else None
    product_col = schema.get("product_column") if schema.get("product_column") in df.columns else None
    customer_col = schema.get("customer_column") if schema.get("customer_column") in df.columns else None

    def _first_existing(*cols: str | None) -> str | None:
        for col in cols:
            if col and col in df.columns:
                return col
        return None

    salesperson_col = _first_existing(
        schema.get("salesperson_column"),
        next((c for c in df.columns if "salesperson" in _normalize_name(c) or "rep" in _normalize_name(c)), None),
    )
    channel_col = _first_existing(
        schema.get("sales_channel_column"),
        next((c for c in df.columns if "channel" in _normalize_name(c)), None),
    )
    campaign_col = _first_existing(
        schema.get("campaign_source_column"),
        next((c for c in df.columns if "campaign" in _normalize_name(c) or "source" in _normalize_name(c)), None),
    )
    warehouse_col = _first_existing(
        schema.get("warehouse_column"),
        next((c for c in df.columns if "warehouse" in _normalize_name(c) or "depot" in _normalize_name(c)), None),
    )
    status_col = _first_existing(
        schema.get("order_status_column"),
        next((c for c in df.columns if "status" in _normalize_name(c)), None),
    )
    brand_col = _first_existing(
        schema.get("product_brand_column"),
        next((c for c in df.columns if "brand" in _normalize_name(c)), None),
    )

    sales_series = _coerce_numeric_series(df, sales_col)
    clean_sales = sales_series.dropna() if sales_series is not None else pd.Series(dtype=float)

    parts = []
    follow_ups = []
    recommendation_notes = []

    margin = metrics.get("profit_margin")
    if margin is not None:
        margin_pct = float(margin) * 100
        if margin_pct < 0:
            parts.append(
                f"Your dataset is currently running at a negative profit margin of {margin_pct:.2f}%. The fastest way to improve sales quality is to cut loss-making volume before scaling more traffic."
            )
        elif margin_pct < 10:
            parts.append(
                f"Your profit margin is only {margin_pct:.2f}%, so you should grow sales with tighter discount control and better product mix."
            )
        else:
            parts.append(
                f"Your profit margin looks healthy at {margin_pct:.2f}%. The best move is to scale the strongest products, regions, and channels."
            )

    if clean_sales.empty:
        return {
            "answer": "I found a sales column, but there are not enough numeric values to analyze it reliably.",
            "intent": "sales_strategy",
            "confidence": 0.5,
        }

    factors = _infer_sales_factors(df, sales_col, profit_col=profit_col, feature_importance=feature_importance)

    if factors["numeric"]:
        top_numeric = factors["numeric"][:3]
        numeric_lines = []
        for name, corr, abs_corr in top_numeric:
            direction = "positive" if corr > 0 else "negative"
            numeric_lines.append(
                f"  - **{name}** has a {direction} relationship with {sales_col} (corr {corr:+.2f})."
            )
        parts.append(
            "Key numeric factors affecting sales:\n" + "\n".join(numeric_lines)
        )

    if factors["categorical"]:
        best_cat = factors["categorical"][0]
        parts.append(
            f"The strongest categorical split is **{best_cat[0]}**. The best group reaches {_fmt(best_cat[1])} in sales while the weakest group is at {_fmt(best_cat[2])}, so this dimension clearly changes performance."
        )

    if factors["model"]:
        model_lines = [
            f"  - **{name}** is one of the strongest model features."
            for name, _ in factors["model"][:3]
        ]
        parts.append(
            "Model feature importance also points to:\n" + "\n".join(model_lines)
        )

    performer_candidates = []
    for label, col in [
        ("product", product_col),
        ("category", schema.get("category_column") if schema.get("category_column") in df.columns else None),
        ("region", region_col),
        ("salesperson", salesperson_col),
        ("channel", channel_col),
        ("campaign", campaign_col),
        ("customer", customer_col),
        ("brand", brand_col),
        ("warehouse", warehouse_col),
        ("status", status_col),
    ]:
        if col:
            performer_candidates.append((label, col))

    performer = _infer_sales_performer(df, sales_col, performer_candidates)
    if performer:
        parts.append(
            f"Best performer by {performer['label']}: **{performer['best_name']}** with {_fmt(performer['best_value'])} in {sales_col}."
        )
        parts.append(
            f"Worst performer by {performer['label']}: **{performer['worst_name']}** with {_fmt(performer['worst_value'])} in {sales_col}."
        )

        performer_agg = _aggregate_sales_by_dimension(df, sales_col, performer["column"])
        concentration_text = _group_concentration_text(
            performer_agg, performer["label"], sales_col
        )
        if concentration_text:
            parts.append(concentration_text)
            recommendation_notes.append(
                f"Scale the strongest {performer['label']}s and use the winning playbook from **{performer['best_name']}**."
            )

    if discount_col:
        discount_series = _coerce_numeric_series(df, discount_col)
        if discount_series is not None:
            paired = pd.concat([clean_sales, discount_series], axis=1).dropna()
            if len(paired) >= 8 and paired.iloc[:, 1].nunique(dropna=True) > 1:
                corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
                if pd.notna(corr):
                    if corr < 0:
                        parts.append(
                            f"Discounts appear to move sales in the opposite direction of margin: larger discounts are associated with weaker sales quality in this dataset (corr {corr:+.2f})."
                        )
                        recommendation_notes.append(
                            "Reduce deep discounts and test tighter discount bands on weak segments."
                        )
                    else:
                        parts.append(
                            f"Discounts are positively associated with sales volume here (corr {corr:+.2f}), but you should still cap them if profit is soft."
                        )
                        recommendation_notes.append(
                            "Use discounts selectively, only where they clearly lift revenue more than they compress profit."
                        )

    if date_col:
        try:
            ts = df.copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, sales_col])
            monthly = ts.set_index(date_col).resample("ME")[sales_col].sum()
            if len(monthly) >= 4:
                recent = monthly.tail(3).mean()
                previous_slice = monthly.iloc[max(0, len(monthly) - 6) : max(0, len(monthly) - 3)]
                previous = previous_slice.mean() if len(previous_slice) > 0 else None
                if previous not in (None, 0) and pd.notna(previous):
                    growth = ((recent - previous) / abs(previous)) * 100
                    if growth < -5:
                        parts.append(
                            f"Recent sales are down {abs(growth):.1f}% versus the prior period, so a short promotion, better campaign targeting, or a channel review would be the fastest fix."
                        )
                        recommendation_notes.append(
                            "Run a short promotion and review the weakest channel or campaign."
                        )
                    elif growth > 5:
                        parts.append(
                            f"Recent sales are up {growth:.1f}% versus the prior period, so the winning tactic is to scale the same offer, channel, and product mix."
                        )
                        recommendation_notes.append(
                            "Scale the same offer, channel, and product mix that drove the recent lift."
                        )
        except Exception:
            pass

    if not parts:
        parts.append(
            "The dataset does not expose a strong sales pattern yet, but the safest technique is to scale the highest-performing segment, reduce discounts on weak segments, and run a controlled promotion test."
        )

    if "increase sales" in q or "improve sales" in q or "boost sales" in q:
        parts.insert(
            0,
            "To increase sales, focus on the segment with the highest sales lift and copy its pricing, channel, and product mix into weaker segments.",
        )
        recommendation_notes.append(
            "Prioritize the strongest segment and replicate its pricing, placement, and mix into weak segments."
        )
    elif "factor" in q or "affect" in q or "impact" in q or "driver" in q:
        parts.insert(
            0,
            "The main factors affecting sales are the columns with the strongest correlations and the widest gaps between top and bottom performers.",
        )
        if factors["numeric"]:
            top_factor = factors["numeric"][0][0]
            recommendation_notes.append(
                f"Focus on the strongest driver, **{_display_column_name(top_factor)}**, and test whether moving it improves {sales_col}."
            )

    if recommendation_notes:
        deduped_notes = []
        seen_notes = set()
        for note in recommendation_notes:
            if note not in seen_notes:
                seen_notes.add(note)
                deduped_notes.append(note)
        recommendation_text = "Recommended technique: " + " ".join(deduped_notes[:2])
        parts.insert(1 if parts else 0, recommendation_text)

    suggested_questions = _build_sales_suggestions(
        question,
        sales_col,
        performer=performer,
        region_col=region_col,
        product_col=product_col,
        customer_col=customer_col,
        salesperson_col=salesperson_col,
        channel_col=channel_col,
        campaign_col=campaign_col,
        warehouse_col=warehouse_col,
        status_col=status_col,
        brand_col=brand_col,
        profit_col=profit_col,
        discount_col=discount_col,
        date_col=date_col,
        top_numeric=factors["numeric"],
    )

    answer = (
        "**Sales analysis**\n"
        + "\n".join(f"  - {part}" for part in parts[:5])
    )

    return {
        "answer": answer,
        "intent": "sales_strategy",
        "confidence": 0.94,
        "suggested_questions": suggested_questions,
    }


def _answer_recommendation(question: str, artifacts: DatasetArtifacts) -> dict:
    return _build_sales_strategy_answer(question, artifacts)


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
        '*"What is the total revenue?"* or *"Show me top 5 products by sales."*'
    )
    return {"answer": capabilities, "intent": "greeting", "confidence": 1.0}


def _answer_recommendation_legacy(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    metrics = artifacts.metrics or {}
    df = artifacts.df
    q = question.lower()

    if df is None:
        return {
            "answer": "I need the cleaned dataset before I can suggest business actions. Please complete cleaning and try again.",
            "intent": "recommendation",
            "confidence": 0.4,
            "suggested_questions": [
                "What columns are available?",
                "What is the total sales?",
                "Are there any missing values?",
            ],
        }

    def resolve_column(role: str, keywords: list[str]) -> str | None:
        schema_key = f"{role}_column"
        if schema.get(schema_key) and schema[schema_key] in df.columns:
            return schema[schema_key]
        for col in df.columns:
            c = str(col).lower()
            if any(k in c for k in keywords):
                return col
        return None

    def numeric_series(col: str | None):
        if not col or col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce")

    sales_col = resolve_column("sales", ["sales", "revenue", "turnover"])
    profit_col = resolve_column("profit", ["profit", "margin", "earnings"])
    region_col = resolve_column("region", ["region", "state", "country", "city", "territory"])
    product_col = resolve_column("product", ["product", "item", "sku", "category", "sub-category", "subcategory"])
    customer_col = resolve_column("customer", ["customer", "client", "buyer"])
    discount_col = resolve_column("discount", ["discount", "rebate", "markdown"])
    date_col = resolve_column("date", ["date", "day", "month", "time", "year"])

    focus_text = None
    focus_match = re.search(
        r"\b(?:improve|increase|boost|grow|fix|help)\s+(?:my\s+|the\s+)?([a-z0-9][a-z0-9 _-]{1,40})",
        q,
    )
    if focus_match:
        focus_text = focus_match.group(1).strip(" ?.!,'\"")

    categorical_cols = [c for c in [region_col, product_col, customer_col] if c and c in df.columns]
    focus_col = None
    focus_value = None
    if focus_text and categorical_cols:
        for col in categorical_cols:
            values = df[col].dropna().astype(str).unique().tolist()
            exact = next((v for v in values if v.lower() == focus_text.lower()), None)
            partial = next((v for v in values if focus_text.lower() in v.lower() or v.lower() in focus_text.lower()), None)
            match = exact or partial
            if match:
                focus_col = col
                focus_value = match
                break

    parts = []
    follow_up_questions = []

    # Focused advice for a specific segment/entity, e.g. "Technology"
    if focus_col and focus_value:
        segment = df[df[focus_col].astype(str).str.lower() == str(focus_value).lower()].copy()
        if not segment.empty:
            parts.append(
                f"To improve **{focus_value}**, focus on the levers that matter most for that segment rather than the whole dataset."
            )

            if sales_col and sales_col in segment.columns:
                seg_sales = numeric_series(sales_col)
                segment["_sales_metric"] = seg_sales
                segment_sales = segment["_sales_metric"].dropna()
                if not segment_sales.empty:
                    overall_sales = numeric_series(sales_col)
                    if overall_sales is not None and overall_sales.dropna().mean() > 0:
                        ratio = segment_sales.mean() / overall_sales.dropna().mean()
                        if ratio < 0.85:
                            parts.append(
                                f"**{focus_value}** is underperforming on sales versus the rest of the dataset. Improve it by reviewing pricing, placement, and product mix for that segment."
                            )
                        else:
                            parts.append(
                                f"**{focus_value}** is holding up reasonably well on sales. The bigger opportunity is to protect margin and scale the winning attributes of this segment."
                            )

            if profit_col and profit_col in segment.columns:
                seg_profit = numeric_series(profit_col)
                segment["_profit_metric"] = seg_profit
                segment_profit = segment["_profit_metric"].dropna()
                if not segment_profit.empty:
                    overall_profit = numeric_series(profit_col)
                    if overall_profit is not None and overall_profit.dropna().mean() is not None:
                        if segment_profit.mean() < overall_profit.dropna().mean():
                            parts.append(
                                f"Profit for **{focus_value}** is below the dataset average, so discount discipline and cost control matter here."
                            )

            if discount_col and discount_col in segment.columns:
                seg_disc = numeric_series(discount_col)
                if seg_disc is not None and seg_disc.dropna().any():
                    parts.append(
                        f"Check discounts on **{focus_value}**. If it is heavily discounted, test whether a smaller discount keeps profit healthier."
                    )

            if region_col and region_col in df.columns and focus_col != region_col:
                region_sales = numeric_series(sales_col)
                if region_sales is not None and region_sales.dropna().any():
                    region_totals = (
                        df.assign(_sales_metric=region_sales)
                        .dropna(subset=[region_col, "_sales_metric"])
                        .groupby(region_col)["_sales_metric"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    if len(region_totals) > 1:
                        best_region = region_totals.idxmax()
                        weakest_region = region_totals.idxmin()
                        parts.append(
                            f"Outside of **{focus_value}**, the strongest region is **{best_region}** and the weakest is **{weakest_region}**. Use the best region as a playbook."
                        )

            follow_up_questions.extend([
                f"What makes {focus_value} perform well?",
                f"How can I improve {focus_value} profit?",
                f"Should I reduce discounts on {focus_value}?",
            ])
        else:
            parts.append(
                f"I found **{focus_value}** in your question, but not as a clean category in the data. I can still help by looking at sales, profit, discounts, and trends overall."
            )

    # Overall business advice
    if not parts:
        margin = metrics.get("profit_margin")
        if margin is not None:
            margin_pct = float(margin) * 100
            if margin_pct < 0:
                parts.append(
                    f"Your data shows a negative profit margin of {margin_pct:.2f}%, so the fastest wins are to reduce loss-making sales, tighten discounting, and review high-cost products."
                )
            elif margin_pct < 10:
                parts.append(
                    f"Your profit margin is only {margin_pct:.2f}%, which suggests there is room to improve pricing, mix, or discounts."
                )
            else:
                parts.append(
                    f"Your profit margin looks healthy at {margin_pct:.2f}%. The next opportunity is to scale the strongest products and regions."
                )

        sales_series = numeric_series(sales_col)
        if sales_series is not None and sales_series.dropna().any():
            working = df.copy()
            working["_sales_metric"] = sales_series

            if region_col and region_col in working.columns:
                region_sales = (
                    working.dropna(subset=[region_col, "_sales_metric"])
                    .groupby(region_col)["_sales_metric"]
                    .sum()
                    .sort_values(ascending=False)
                )
                if len(region_sales) > 1:
                    best_region = region_sales.idxmax()
                    weakest_region = region_sales.idxmin()
                    parts.append(
                        f"Sales are strongest in **{best_region}** and weakest in **{weakest_region}**. Focus promotions, stock, or outreach on the weaker region."
                    )
                    follow_up_questions.extend([
                        f"Why is {weakest_region} underperforming?",
                        f"How can I grow sales in {weakest_region}?",
                    ])

            if product_col and product_col in working.columns:
                product_sales = (
                    working.dropna(subset=[product_col, "_sales_metric"])
                    .groupby(product_col)["_sales_metric"]
                    .sum()
                    .sort_values(ascending=False)
                )
                if len(product_sales) > 1:
                    best_product = product_sales.idxmax()
                    weakest_product = product_sales.idxmin()
                    parts.append(
                        f"Your best product or category is **{best_product}**, while **{weakest_product}** is the weakest. Double down on the winner and review pricing, bundling, or placement for the weaker one."
                    )
                    follow_up_questions.extend([
                        f"What makes {best_product} perform well?",
                        f"How can I improve {weakest_product}?",
                    ])

            if date_col and date_col in working.columns and len(working) >= 6:
                try:
                    ts = working.copy()
                    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                    ts = ts.dropna(subset=[date_col, "_sales_metric"])
                    monthly = ts.set_index(date_col).resample("ME")["_sales_metric"].sum()
                    if len(monthly) >= 4:
                        recent = monthly.tail(3).mean()
                        previous_slice = monthly.iloc[max(0, len(monthly) - 6): max(0, len(monthly) - 3)]
                        previous = previous_slice.mean() if len(previous_slice) > 0 else None
                        if previous not in (None, 0) and pd.notna(previous):
                            change = ((recent - previous) / abs(previous)) * 100
                            if change < -5:
                                parts.append(
                                    f"Recent sales are down {abs(change):.1f}% versus the previous period. You may want to run a promotion, adjust pricing, or revisit channel performance."
                                )
                                follow_up_questions.extend([
                                    "What is the monthly sales trend?",
                                    "Which month performed the worst?",
                                ])
                except Exception:
                    pass

        if profit_col and profit_col in df.columns and discount_col and discount_col in df.columns:
            try:
                tmp = df[[discount_col, profit_col]].copy()
                tmp[discount_col] = pd.to_numeric(tmp[discount_col], errors="coerce")
                tmp[profit_col] = pd.to_numeric(tmp[profit_col], errors="coerce")
                tmp = tmp.dropna()
                if len(tmp) >= 10 and tmp[discount_col].nunique() > 1:
                    low_discount = tmp[tmp[discount_col] <= tmp[discount_col].quantile(0.25)][profit_col].mean()
                    high_discount = tmp[tmp[discount_col] >= tmp[discount_col].quantile(0.75)][profit_col].mean()
                    if pd.notna(low_discount) and pd.notna(high_discount) and high_discount < low_discount:
                        parts.append(
                            "Higher discounts appear to be reducing profit. Try limiting deep discounts to only the slowest-moving items."
                        )
                        follow_up_questions.extend([
                            "What discount level is safest for profit?",
                            "Which products should not be discounted?",
                        ])
            except Exception:
                pass

        if customer_col and customer_col in df.columns and sales_col and sales_col in df.columns:
            customer_sales = (
                df.dropna(subset=[customer_col])
                .assign(_sales_metric=numeric_series(sales_col))
                .dropna(subset=["_sales_metric"])
                .groupby(customer_col)["_sales_metric"]
                .sum()
                .sort_values(ascending=False)
            )
            if len(customer_sales) > 1:
                top_customer = customer_sales.idxmax()
                parts.append(
                    f"Your highest-value customer is **{top_customer}**. A retention or upsell plan for top customers can improve overall revenue faster than broad campaigns."
                )
                follow_up_questions.extend([
                    "Which customers generate the most revenue?",
                    f"How can I retain {top_customer}?",
                ])

    if not parts:
        parts.append(
            "The dataset does not expose enough business signals for a strong recommendation yet, but a good next step is to identify the highest-value segment and the biggest loss driver."
        )

    if "sales" in q or "revenue" in q:
        parts.insert(0, "To improve sales and revenue, focus on the biggest controllable levers in the dataset.")
        follow_up_questions.extend([
            "Which product or category should I scale?",
            "Where are discounts hurting profit?",
        ])
    elif "profit" in q or "margin" in q:
        parts.insert(0, "To improve profit, reduce leakage before chasing more volume.")
        follow_up_questions.extend([
            "Which products have the best profit margin?",
            "How do discounts affect profit?",
        ])

    follow_up_questions.extend([
        "What are the top 5 products by sales?",
        "Which region is underperforming?",
        "What are the monthly trends?",
        "Are there any anomalies?",
    ])

    deduped_questions = []
    seen = set()
    for item in follow_up_questions:
        if item not in seen:
            seen.add(item)
            deduped_questions.append(item)

    answer = (
        "**Recommended actions:**\n"
        + "\n".join(f"  • {part}" for part in parts[:4])
        + "\n\n**Good follow-up questions:**\n"
        + "\n".join(f"  • {q}" for q in deduped_questions[:5])
    )
    return {
        "answer": answer,
        "intent": "recommendation",
        "confidence": 0.92,
        "suggested_questions": deduped_questions[:5],
    }


def _answer_summary(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    kpi = artifacts.kpi
    schema = artifacts.schema
    df = artifacts.df

    parts = []

    if profile:
        rows = profile.get("row_count", "?")
        cols = profile.get("column_count", "?")
        parts.append(f"**Dataset Overview:** {rows:,} rows × {cols} columns.")

        num_cols = profile.get("numeric_columns", [])
        cat_cols = profile.get("categorical_columns", [])
        if num_cols:
            parts.append(
                f"**Numeric columns:** {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}."
            )
        if cat_cols:
            parts.append(
                f"**Categorical columns:** {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}."
            )

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

    if not parts and df is not None:
        rows = len(df)
        cols = len(df.columns)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        parts.append(f"**Dataset Overview:** {rows:,} rows × {cols} columns.")

        if num_cols:
            parts.append(
                f"**Numeric columns:** {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}."
            )

        if cat_cols:
            parts.append(
                f"**Categorical columns:** {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}."
            )

        if missing.empty:
            parts.append("**No missing values** found in the loaded dataset. ✅")
        else:
            mv_str = ", ".join(
                f"{col}: {int(val)}" for col, val in missing.head(5).items()
            )
            parts.append(f"**Missing values detected in:** {mv_str}.")

        sales_col = schema.get("sales_column")
        if sales_col and sales_col in df.columns and _is_numeric_column(df, sales_col):
            parts.append(f"**Total {sales_col}:** {_fmt(df[sales_col].sum())}.")

        profit_col = schema.get("profit_column")
        if profit_col and profit_col in df.columns and _is_numeric_column(df, profit_col):
            parts.append(f"**Total {profit_col}:** {_fmt(df[profit_col].sum())}.")

    if not parts:
        return {
            "answer": "I could not build a summary yet. Please make sure the dataset has been loaded and processed.",
            "intent": "summary",
            "confidence": 0.5,
        }

    return {"answer": "\n\n".join(parts), "intent": "summary", "confidence": 0.9}


def _answer_advanced_mode(artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    if df is None:
        return {
            "answer": "The dataset is not loaded yet. Please wait for processing to complete.",
            "intent": "advanced_mode",
            "confidence": 0.4,
        }

    sales_col = schema.get("sales_column")
    profit_col = schema.get("profit_column")
    region_col = schema.get("region_column")
    product_col = schema.get("product_column")
    customer_col = schema.get("customer_column")
    date_col = schema.get("date_column")

    parts = [
        f"**Advanced dataset mode is ready** for **{len(df):,} rows** and **{len(df.columns)} columns**."
    ]

    if sales_col and sales_col in df.columns and _is_numeric_column(df, sales_col):
        parts.append(f"**Main sales metric:** {sales_col} with total {_fmt(df[sales_col].sum())}.")

    if profit_col and profit_col in df.columns and _is_numeric_column(df, profit_col):
        parts.append(f"**Profit metric:** {profit_col} with total {_fmt(df[profit_col].sum())}.")

    roles = [
        label
        for label, col in [
            ("region", region_col),
            ("product", product_col),
            ("customer", customer_col),
            ("date", date_col),
        ]
        if col and col in df.columns
    ]
    if roles:
        parts.append(f"**Detected analysis dimensions:** {', '.join(roles)}.")

    parts.append(
        "Ask me about totals, top and bottom performers, trends, profit drivers, "
        "or revenue improvement ideas."
    )

    return {
        "answer": "\n\n".join(parts),
        "intent": "advanced_mode",
        "confidence": 0.85,
        "suggested_questions": [
            "What is the total sales?",
            "Which product is the best performer?",
            "Which region is underperforming?",
            "What are the monthly trends?",
            "How can I increase revenue?",
        ],
    }


def _answer_columns(artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    profile = artifacts.profile

    known_roles = {
        k.replace("_column", ""): v
        for k, v in schema.items()
        if k.endswith("_column") and isinstance(v, str)
    }

    if not known_roles and not profile:
        return {
            "answer": "Column information is not available yet. Please ensure the dataset has been processed.",
            "intent": "columns",
            "confidence": 0.5,
        }

    parts = []

    if known_roles:
        role_lines = "\n".join(
            f"  • **{role.capitalize()}** → `{col}`"
            for role, col in known_roles.items()
        )
        parts.append(f"**Detected column roles:**\n{role_lines}")

    if profile:
        num_cols = profile.get("numeric_columns", [])
        cat_cols = profile.get("categorical_columns", [])
        dt_cols = profile.get("datetime_columns", [])
        if num_cols:
            parts.append(
                f"**Numeric columns ({len(num_cols)}):** {', '.join(num_cols)}."
            )
        if cat_cols:
            parts.append(
                f"**Text/Category columns ({len(cat_cols)}):** {', '.join(cat_cols)}."
            )
        if dt_cols:
            parts.append(f"**Date columns ({len(dt_cols)}):** {', '.join(dt_cols)}.")

    return {"answer": "\n\n".join(parts), "intent": "columns", "confidence": 0.9}


def _answer_rows(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile and "row_count" in profile:
        r = profile["row_count"]
        c = profile.get("column_count", "?")
        return {
            "answer": f"The dataset contains **{r:,} rows** and **{c} columns**.",
            "intent": "rows",
            "confidence": 1.0,
        }
    df = artifacts.df
    if df is not None:
        return {
            "answer": f"The dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**.",
            "intent": "rows",
            "confidence": 0.9,
        }
    return {
        "answer": "Row count is not available yet.",
        "intent": "rows",
        "confidence": 0.4,
    }


def _answer_missing(artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile:
        missing = profile.get("missing_values", {})
        if not missing:
            return {
                "answer": "✅ **No missing values** were detected in your dataset.",
                "intent": "missing",
                "confidence": 1.0,
            }
        lines = "\n".join(
            f"  • **{c}**: {v} missing value(s)" for c, v in missing.items()
        )
        return {
            "answer": f"**Missing values detected:**\n{lines}",
            "intent": "missing",
            "confidence": 1.0,
        }
    df = artifacts.df
    if df is not None:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return {
                "answer": "✅ **No missing values** were detected in your dataset.",
                "intent": "missing",
                "confidence": 0.9,
            }
        lines = "\n".join(f"  • **{c}**: {v}" for c, v in missing.items())
        return {
            "answer": f"**Missing values detected:**\n{lines}",
            "intent": "missing",
            "confidence": 0.9,
        }
    return {
        "answer": "Missing value information is not available.",
        "intent": "missing",
        "confidence": 0.4,
    }


def _answer_total(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi = artifacts.kpi

    # Determine target column from question
    target_col = _resolve_metric_column(question, schema, artifacts.df) or _find_target_column(
        question, schema, artifacts.df
    )

    # Try KPI cache first
    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("sum") or kpi[target_col].get("total")
        if val is not None:
            return {
                "answer": f"The **total {target_col}** is **{_fmt(val)}**.",
                "intent": "total_sales"
                if "sales" in question.lower() or "revenue" in question.lower()
                else "total_generic",
                "confidence": 0.95,
            }

    # Fallback: compute from CSV
    df = artifacts.df
    if df is not None and target_col and target_col in df.columns and _is_numeric_column(df, target_col):
        val = df[target_col].sum()
        return {
            "answer": f"The **total {target_col}** is **{_fmt(val)}**.",
            "intent": "total_generic",
            "confidence": 0.85,
        }

    # Try sales column directly
    sales_col = schema.get("sales_column")
    if sales_col and kpi and sales_col in kpi:
        val = kpi[sales_col].get("sum") or kpi[sales_col].get("total")
        if val is not None:
            return {
                "answer": f"The **total {sales_col}** is **{_fmt(val)}**.",
                "intent": "total_sales",
                "confidence": 0.9,
            }

    if df is not None and sales_col and sales_col in df.columns:
        val = df[sales_col].sum()
        return {
            "answer": f"The **total {sales_col}** is **{_fmt(val)}**.",
            "intent": "total_sales",
            "confidence": 0.8,
        }

    return {
        "answer": "I could not compute a total from your dataset. Please ensure the dataset has been processed successfully.",
        "intent": "total_generic",
        "confidence": 0.3,
    }


def _answer_average(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi = artifacts.kpi
    target_col = _resolve_metric_column(question, schema, artifacts.df) or _find_target_column(
        question, schema, artifacts.df
    )

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("mean") or kpi[target_col].get("avg")
        if val is not None:
            return {
                "answer": f"The **average {target_col}** is **{_fmt(val)}**.",
                "intent": "average",
                "confidence": 0.95,
            }

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns and _is_numeric_column(df, target_col):
        val = df[target_col].mean()
        return {
            "answer": f"The **average {target_col}** is **{_fmt(val)}**.",
            "intent": "average",
            "confidence": 0.85,
        }

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].mean()
        return {
            "answer": f"The **average {sales_col}** per record is **{_fmt(val)}**.",
            "intent": "average",
            "confidence": 0.75,
        }

    return {
        "answer": "I could not compute an average. Please check the dataset has been processed.",
        "intent": "average",
        "confidence": 0.3,
    }


def _answer_max(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi = artifacts.kpi
    target_col = _resolve_metric_column(question, schema, artifacts.df) or _find_target_column(
        question, schema, artifacts.df
    )

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("max")
        if val is not None:
            return {
                "answer": f"The **maximum {target_col}** is **{_fmt(val)}**.",
                "intent": "maximum",
                "confidence": 0.95,
            }

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns and _is_numeric_column(df, target_col):
        val = df[target_col].max()
        return {
            "answer": f"The **maximum {target_col}** is **{_fmt(val)}**.",
            "intent": "maximum",
            "confidence": 0.85,
        }

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].max()
        return {
            "answer": f"The **highest {sales_col}** in a single record is **{_fmt(val)}**.",
            "intent": "maximum",
            "confidence": 0.75,
        }

    return {
        "answer": "I could not find the maximum value.",
        "intent": "maximum",
        "confidence": 0.3,
    }


def _answer_min(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi = artifacts.kpi
    target_col = _resolve_metric_column(question, schema, artifacts.df) or _find_target_column(
        question, schema, artifacts.df
    )

    if target_col and kpi and target_col in kpi:
        val = kpi[target_col].get("min")
        if val is not None:
            return {
                "answer": f"The **minimum {target_col}** is **{_fmt(val)}**.",
                "intent": "minimum",
                "confidence": 0.95,
            }

    df = artifacts.df
    if df is not None and target_col and target_col in df.columns and _is_numeric_column(df, target_col):
        val = df[target_col].min()
        return {
            "answer": f"The **minimum {target_col}** is **{_fmt(val)}**.",
            "intent": "minimum",
            "confidence": 0.85,
        }

    sales_col = schema.get("sales_column")
    if sales_col and df is not None and sales_col in df.columns:
        val = df[sales_col].min()
        return {
            "answer": f"The **lowest {sales_col}** in a single record is **{_fmt(val)}**.",
            "intent": "minimum",
            "confidence": 0.75,
        }

    return {
        "answer": "I could not find the minimum value.",
        "intent": "minimum",
        "confidence": 0.3,
    }


def _answer_count(question: str, artifacts: DatasetArtifacts) -> dict:
    profile = artifacts.profile
    if profile and "row_count" in profile:
        val = profile["row_count"]
        return {
            "answer": f"Your dataset has **{val:,} records** (rows).",
            "intent": "count",
            "confidence": 1.0,
        }
    df = artifacts.df
    if df is not None:
        return {
            "answer": f"Your dataset has **{len(df):,} records** (rows).",
            "intent": "count",
            "confidence": 0.9,
        }
    return {
        "answer": "Record count is not available.",
        "intent": "count",
        "confidence": 0.3,
    }


def _answer_top_n(question: str, artifacts: DatasetArtifacts) -> dict:
    n = _extract_n(question)
    schema = artifacts.schema
    df = artifacts.df

    if df is None:
        return {
            "answer": "Dataset not yet available for analysis.",
            "intent": "top_n",
            "confidence": 0.3,
        }

    metric_col = _resolve_metric_column(question, schema, df)
    group_col, group_label = _resolve_group_column(question, schema, df, metric_col=metric_col)
    val_col = metric_col or schema.get("sales_column") or schema.get("profit_column")

    if (
        not group_col
        or not val_col
        or group_col not in df.columns
        or val_col not in df.columns
        or not _is_numeric_column(df, val_col)
    ):
        return {
            "answer": f"I could not determine a grouping or value column to compute top {n}.",
            "intent": "top_n",
            "confidence": 0.4,
        }

    is_worst = bool(re.search(r"\bworst\b|\blowest\b|\bbottom\b", question.lower()))

    agg = df.groupby(group_col)[val_col].sum().sort_values(ascending=is_worst)
    top = agg.head(n)

    label = "bottom" if is_worst else "top"
    lines = "\n".join(
        f"  {i + 1}. **{name}** — {_fmt(val)}"
        for i, (name, val) in enumerate(top.items())
    )
    return {
        "answer": f"**{label.capitalize()} {n} {group_label} by {val_col}:**\n{lines}",
        "intent": "top_n",
        "confidence": 0.9,
    }


def _answer_driver(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    if df is None:
        return {
            "answer": "Dataset not yet available for analysis.",
            "intent": "driver",
            "confidence": 0.3,
        }

    metric_col = _resolve_metric_column(question, schema, df)
    if not metric_col or metric_col not in df.columns or not _is_numeric_column(df, metric_col):
        metric_col = schema.get("sales_column") if schema.get("sales_column") in df.columns else None

    if not metric_col or not _is_numeric_column(df, metric_col):
        return {
            "answer": "I could not find a numeric sales column to rank the drivers.",
            "intent": "driver",
            "confidence": 0.4,
        }

    group_col, group_label = _resolve_group_column(question, schema, df, metric_col=metric_col)
    if not group_col or group_col not in df.columns:
        return {
            "answer": "I could not determine which product or category to rank.",
            "intent": "driver",
            "confidence": 0.4,
        }

    agg = (
        df.dropna(subset=[group_col])
        .assign(_metric=pd.to_numeric(df[metric_col], errors="coerce"))
        .dropna(subset=["_metric"])
        .groupby(group_col)["_metric"]
        .sum()
        .sort_values(ascending=False)
    )

    if agg.empty:
        return {
            "answer": "I could not compute a driver ranking from the available data.",
            "intent": "driver",
            "confidence": 0.4,
        }

    best_name = agg.idxmax()
    best_val = agg.iloc[0]
    top5 = agg.head(5)
    lines = "\n".join(
        f"  {i + 1}. **{name}** - {_fmt(val)}"
        for i, (name, val) in enumerate(top5.items())
    )

    answer = (
        f"**Top sales driver by {group_label}:** **{best_name}** with {_fmt(best_val)} in {metric_col}.\n\n"
        f"**Top 5 {group_label} by {metric_col}:**\n{lines}"
    )
    return {"answer": answer, "intent": "driver", "confidence": 0.95}


def _answer_region(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    val_col = _resolve_metric_column(question, schema, df) or schema.get("sales_column") or schema.get("profit_column")
    region_col, _ = _resolve_group_column(question, schema, df, metric_col=val_col)
    if region_col and "region" not in _normalize_name(region_col):
        region_col = schema.get("region_column")

    if not region_col or df is None:
        return {
            "answer": "Region column could not be detected in your dataset.",
            "intent": "region",
            "confidence": 0.4,
        }

    if val_col not in df.columns or not _is_numeric_column(df, val_col):
        return {
            "answer": f"Could not find a suitable value column to aggregate by region.",
            "intent": "region",
            "confidence": 0.4,
        }

    agg = df.dropna(subset=[region_col]).groupby(region_col)[val_col].sum().sort_values(ascending=False)
    best = agg.idxmax()
    worst = agg.idxmin()

    top5 = agg.head(5)
    lines = "\n".join(
        f"  {i + 1}. **{r}** — {_fmt(v)}" for i, (r, v) in enumerate(top5.items())
    )

    answer = (
        f"**Regional breakdown by {val_col}:**\n{lines}\n\n"
        f"🏆 **Best region:** {best} ({_fmt(agg[best])})\n"
        f"⚠️ **Weakest region:** {worst} ({_fmt(agg[worst])})"
    )
    return {"answer": answer, "intent": "region", "confidence": 0.9}


def _answer_product(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    val_col = _resolve_metric_column(question, schema, df) or schema.get("sales_column")
    product_col, group_label = _resolve_group_column(question, schema, df, metric_col=val_col)
    if product_col and "product" not in _normalize_name(product_col) and "category" not in _normalize_name(product_col):
        product_col = schema.get("product_column")

    if not product_col or df is None:
        return _answer_top_n(question + " by product", artifacts)

    if val_col not in df.columns or not _is_numeric_column(df, val_col):
        return {
            "answer": "Could not find a numeric column to rank products by.",
            "intent": "product",
            "confidence": 0.4,
        }

    agg = df.dropna(subset=[product_col]).groupby(product_col)[val_col].sum().sort_values(ascending=False)
    best = agg.idxmax()
    worst = agg.idxmin()

    top5 = agg.head(5)
    lines = "\n".join(
        f"  {i + 1}. **{p}** — {_fmt(v)}" for i, (p, v) in enumerate(top5.items())
    )

    metric_name = _display_column_name(val_col)
    metric_lower = metric_name.lower()
    if any(term in metric_lower for term in ("profit", "margin")):
        best_label = "Most profitable"
        worst_label = "Least profitable"
    elif any(term in metric_lower for term in ("sales", "revenue", "turnover", "income")):
        best_label = "Best-selling"
        worst_label = "Lowest selling"
    else:
        best_label = "Best performing"
        worst_label = "Worst performing"

    answer = (
        f"**Top {group_label} by {metric_name}:**\n{lines}\n\n"
        f"🏆 **{best_label}:** {best} ({_fmt(agg[best])})\n"
        f"⚠️ **{worst_label}:** {worst} ({_fmt(agg[worst])})"
    )
    return {"answer": answer, "intent": "product", "confidence": 0.9}


def _answer_customer(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    val_col = _resolve_metric_column(question, schema, df) or schema.get("sales_column")
    cust_col, group_label = _resolve_group_column(question, schema, df, metric_col=val_col)
    if cust_col and "customer" not in _normalize_name(cust_col):
        cust_col = schema.get("customer_column")

    if not cust_col or df is None:
        return {
            "answer": "Customer column could not be detected in your dataset.",
            "intent": "customer",
            "confidence": 0.4,
        }

    if val_col not in df.columns or not _is_numeric_column(df, val_col):
        return {
            "answer": "Could not find a numeric column to rank customers by.",
            "intent": "customer",
            "confidence": 0.4,
        }

    n = _extract_n(question, default=5)
    agg = df.dropna(subset=[cust_col]).groupby(cust_col)[val_col].sum().sort_values(ascending=False)
    top = agg.head(n)

    lines = "\n".join(
        f"  {i + 1}. **{c}** — {_fmt(v)}" for i, (c, v) in enumerate(top.items())
    )
    total_customers = df[cust_col].nunique()

    answer = f"**Top {n} customers by {val_col}:**\n{lines}\n\n📊 Total unique customers: **{total_customers:,}**"
    return {"answer": answer, "intent": "customer", "confidence": 0.9}


def _answer_profit(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    kpi = artifacts.kpi
    metrics = artifacts.metrics
    df = artifacts.df

    profit_col = _resolve_metric_column(question, schema, df)
    if not profit_col or not _is_numeric_column(df, profit_col):
        profit_col = schema.get("profit_column") if df is not None and schema.get("profit_column") in df.columns else None
    if not profit_col and df is not None:
        profit_col = next(
            (
                c
                for c in df.columns
                if _is_numeric_column(df, c)
                and any(term in _normalize_name(c) for term in ["profit", "margin", "earnings", "income"])
            ),
            None,
        )
    sales_col = schema.get("sales_column") if df is not None and schema.get("sales_column") in df.columns else None
    if not sales_col and df is not None:
        sales_col = next(
            (
                c
                for c in df.columns
                if _is_numeric_column(df, c)
                and any(term in _normalize_name(c) for term in ["sales", "revenue", "turnover", "gross", "net"])
            ),
            None,
        )

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

    if df is not None and profit_col and sales_col and profit_col in df.columns and sales_col in df.columns:
        profit_series = pd.to_numeric(df[profit_col], errors="coerce")
        sales_series = pd.to_numeric(df[sales_col], errors="coerce")
        total_profit = profit_series.sum()
        total_sales = sales_series.sum()
        if pd.notna(total_sales) and float(total_sales) != 0:
            margin = total_profit / total_sales
            if not any("profit margin" in part.lower() for part in parts):
                parts.insert(0, f"**Profit margin:** {round(float(margin) * 100, 2)}%")
            parts.append(
                "Recommended technique: focus on the highest-margin products or regions, reduce deep discounts on weak segments, and scale the playbook from the strongest performer."
            )
        if any(term in question.lower() for term in ["affect", "impact", "influence", "relationship", "driver"]):
            paired = pd.concat([profit_series, sales_series], axis=1).dropna()
            if len(paired) >= 8 and paired.iloc[:, 0].nunique() > 1 and paired.iloc[:, 1].nunique() > 1:
                corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
                if pd.notna(corr):
                    parts.insert(0, f"**Profit vs Sales correlation:** {corr:+.2f}")

    # Compute from df
    if df is not None and profit_col and profit_col in df.columns:
        if not parts:
            total = df[profit_col].sum()
            avg = df[profit_col].mean()
            max_p = df[profit_col].max()
            parts.append(f"**Total {profit_col}:** {_fmt(total)}")
            parts.append(f"**Average {profit_col}:** {_fmt(avg)}")
            parts.append(f"**Best single record:** {_fmt(max_p)}")
    elif df is not None and sales_col and sales_col in df.columns:
        # Estimate if no profit col
        cost_col = schema.get("cost_column") or schema.get("price_column")
        if cost_col and cost_col in df.columns:
            est_profit = (df[sales_col] - df[cost_col]).sum()
            parts.append(
                f"**Estimated total profit** (sales − cost): {_fmt(est_profit)}"
            )

    if not parts:
        return {
            "answer": "Profit data could not be found. The dataset may not have a profit or cost column.",
            "intent": "profit",
            "confidence": 0.4,
        }

    return {"answer": "\n\n".join(parts), "intent": "profit", "confidence": 0.9}


def _answer_loss(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df
    metrics = artifacts.metrics

    profit_col = schema.get("profit_column")

    if metrics:
        margin = metrics.get("profit_margin")
        if margin is not None and float(margin) < 0:
            return {
                "answer": f"⚠️ The dataset shows a **net loss** with a profit margin of **{round(float(margin) * 100, 2)}%**.",
                "intent": "loss",
                "confidence": 0.95,
            }
        if margin is not None and float(margin) >= 0:
            return {
                "answer": f"✅ The dataset is **profitable** with a margin of **{round(float(margin) * 100, 2)}%**. No net loss detected.",
                "intent": "loss",
                "confidence": 0.95,
            }

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
            "confidence": 0.9,
        }

    return {
        "answer": "Loss data could not be computed. No profit column was detected.",
        "intent": "loss",
        "confidence": 0.4,
    }


def _answer_trend(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    date_col = schema.get("date_column")
    sales_col = _resolve_metric_column(question, schema, df) or schema.get("sales_column")
    if sales_col == date_col:
        sales_col = schema.get("sales_column")

    if not date_col or df is None:
        return {
            "answer": "A date column is required for trend analysis, but none was detected.",
            "intent": "trend",
            "confidence": 0.5,
        }

    if date_col not in df.columns:
        return {
            "answer": f"The detected date column '{date_col}' is not present in the data.",
            "intent": "trend",
            "confidence": 0.4,
        }

    if not sales_col or sales_col not in df.columns or not _is_numeric_column(df, sales_col):
        return {
            "answer": "I could not find a numeric sales or revenue column for the trend analysis.",
            "intent": "trend",
            "confidence": 0.4,
        }

    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])

        if re.search(r"\byear\b|\bannual\b", question.lower()):
            freq, label = "YE", "yearly"
        elif re.search(r"\bquarter\b|\bq[1-4]\b", question.lower()):
            freq, label = "QE", "quarterly"
        else:
            freq, label = "ME", "monthly"

        monthly = ts.set_index(date_col).resample(freq)[sales_col].sum()

        if monthly.empty:
            return {
                "answer": "Not enough date data to compute trends.",
                "intent": "trend",
                "confidence": 0.5,
            }

        lines = "\n".join(
            f"  • **{str(idx)[:7]}**: {_fmt(val)}"
            for idx, val in monthly.tail(12).items()
        )

        # Growth rate
        if len(monthly) >= 2:
            first = monthly.iloc[0]
            last = monthly.iloc[-1]
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
            "confidence": 0.9,
        }

    except Exception as e:
        logger.warning(f"Trend calculation failed: {e}")
        return {
            "answer": "I encountered an error computing the trend. Please verify the date column format.",
            "intent": "trend",
            "confidence": 0.4,
        }


def _answer_forecast(question: str, artifacts: DatasetArtifacts) -> dict:
    schema = artifacts.schema
    df = artifacts.df

    date_col = schema.get("date_column")
    sales_col = schema.get("sales_column")

    if not date_col or not sales_col or df is None:
        return {
            "answer": "Forecasting requires both a date column and a sales/revenue column, which could not be detected.",
            "intent": "forecast",
            "confidence": 0.4,
        }

    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col, sales_col])
        monthly = ts.set_index(date_col).resample("ME")[sales_col].sum()

        if len(monthly) < 2:
            return {
                "answer": "Not enough historical data points to make a forecast.",
                "intent": "forecast",
                "confidence": 0.5,
            }

        # Simple moving average forecast
        growth_rate = monthly.pct_change().mean()
        last_val = monthly.iloc[-1]
        next_val = last_val * (1 + growth_rate)

        last_month = monthly.index[-1]
        # Next month
        next_month = last_month + pd.DateOffset(months=1)

        direction = "📈 increase" if growth_rate > 0 else "📉 decrease"

        answer = (
            f"**Forecast for {str(next_month)[:7]}:**\n\n"
            f"  • Predicted {sales_col}: **{_fmt(next_val)}**\n"
            f"  • Based on average monthly growth rate of **{growth_rate * 100:+.1f}%**\n"
            f"  • Last recorded ({str(last_month)[:7]}): {_fmt(last_val)}\n\n"
            f"Trend suggests a {direction} in the next period."
        )
        return {"answer": answer, "intent": "forecast", "confidence": 0.8}

    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
        return {
            "answer": "I encountered an error while computing the forecast.",
            "intent": "forecast",
            "confidence": 0.4,
        }


def _answer_anomaly(artifacts: DatasetArtifacts) -> dict:
    insights_data = artifacts.insights

    if isinstance(insights_data, dict):
        all_insights = insights_data.get("insights", [])
    elif isinstance(insights_data, list):
        all_insights = insights_data
    else:
        all_insights = []

    anomaly_insights = [
        i
        for i in all_insights
        if "anomal" in i.get("type", "").lower()
        or "outlier" in i.get("description", "").lower()
    ]

    if anomaly_insights:
        lines = "\n".join(f"  ⚠️ {i['description']}" for i in anomaly_insights)
        return {
            "answer": f"**Anomalies detected:**\n{lines}",
            "intent": "anomaly",
            "confidence": 0.95,
        }

    # Compute from dataset
    schema = artifacts.schema
    df = artifacts.df
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
                    "confidence": 0.85,
                }
            else:
                return {
                    "answer": f"✅ **No significant anomalies** detected in `{sales_col}`. All values are within 3 standard deviations of the mean.",
                    "intent": "anomaly",
                    "confidence": 0.85,
                }
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

    if not all_insights:
        return {
            "answer": "No anomaly analysis is available yet. Please ensure the dataset pipeline has completed successfully.",
            "intent": "anomaly",
            "confidence": 0.4,
        }

    return {
        "answer": "✅ No anomalies were detected by the AI analysis engine.",
        "intent": "anomaly",
        "confidence": 0.8,
    }


def _answer_insights(artifacts: DatasetArtifacts) -> dict:
    insights_data = artifacts.insights

    if isinstance(insights_data, dict):
        all_insights = insights_data.get("insights", [])
        summary = insights_data.get("summary", "")
    elif isinstance(insights_data, list):
        all_insights = insights_data
        summary = ""
    else:
        all_insights = []
        summary = ""

    if not all_insights and not summary:
        return {
            "answer": "AI insights are not yet available. Please ensure the dataset pipeline completed successfully.",
            "intent": "insight",
            "confidence": 0.5,
        }

    parts = []
    if summary:
        parts.append(f"**Summary:** {summary}")

    icon_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    for i in all_insights[:8]:
        sev = i.get("severity", "info")
        icon = icon_map.get(sev, "ℹ️")
        parts.append(f"{icon} {i.get('description', '')}")

    return {"answer": "\n\n".join(parts), "intent": "insight", "confidence": 0.95}


def _answer_comparison(question: str, artifacts: DatasetArtifacts) -> dict:
    """Try to compare two named entities from the question."""
    schema = artifacts.schema
    df = artifacts.df

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
                        "confidence": 0.85,
                    }

    return {
        "answer": "Please specify two items to compare, e.g., *\"Compare 'North' vs 'South' region\"*.",
        "intent": "comparison",
        "confidence": 0.5,
    }


# ─── Helper: Find Target Column ───────────────────────────────────────────────


def _find_target_column(
    question: str, schema: dict, df: "pd.DataFrame | None"
) -> str | None:
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
            "confidence": 0.0,
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
        elif intent == "driver":
            return _answer_driver(question, artifacts)
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
        elif intent == "advanced_mode":
            return _answer_advanced_mode(artifacts)
        elif intent == "sales_strategy":
            return _answer_recommendation(question, artifacts)
        elif intent == "recommendation":
            return _answer_recommendation(question, artifacts)
        else:
            # Unknown: try a broad keyword search across KPI data
            kpi = artifacts.kpi
            if kpi:
                schema = artifacts.schema
                target = _find_target_column(question, schema, artifacts.df)
                if target and target in kpi:
                    k = kpi[target]
                    lines = "\n".join(
                        f"  • **{stat}**: {_fmt(val)}"
                        for stat, val in k.items()
                        if stat not in ("column",)
                    )
                    return {
                        "answer": f"**Statistics for {target}:**\n{lines}",
                        "intent": "stats",
                        "confidence": 0.7,
                    }

            return {
                "answer": (
                    "I'm not sure how to answer that. Here are some things you can ask:\n\n"
                    '• *"What is the total sales?"*\n'
                    '• *"Show me the top 5 products"*\n'
                    '• *"What are the monthly trends?"*\n'
                    '• *"Give me an overview"*\n'
                    '• *"Are there any anomalies?"*'
                ),
                "intent": "unknown",
                "confidence": 0.0,
            }
    except Exception as e:
        logger.error(
            f"Answer generation failed for intent '{intent}': {e}", exc_info=True
        )
        return {
            "answer": "I encountered an internal error while processing your question. Please try rephrasing it.",
            "intent": intent,
            "confidence": 0.0,
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Dataset Q&A Query Engine")
    parser.add_argument("--user_id", default="default_user")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Override dataset artifact directory (optional)",
    )
    parser.add_argument(
        "--csv_file",
        default=None,
        help="Explicit path to the cleaned CSV file to use for analysis",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or resolve_dataset_dir(args.user_id, args.dataset_id)

    # Pass the explicit csv_file path directly to DatasetArtifacts
    artifacts = DatasetArtifacts(dataset_dir, csv_file=args.csv_file)
    result = answer_question(args.question, artifacts)

    # Single JSON line to stdout — Node.js parses this
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
