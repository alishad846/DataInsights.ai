import os
import json
from filelock import FileLock
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("system_logger")


def generate_dashboard_config(dataset_dir):
    """
    Generates dashboard_config.json with 7 charts covering all major
    analytical views: trend, product, region, customer, quantity, features,
    distribution, and scatter performance.
    """
    schema_path     = os.path.join(dataset_dir, "schema.json")
    clean_path      = os.path.join(dataset_dir, "cleaned_data.csv")
    metrics_def_path = os.path.join(dataset_dir, "metrics_definition.json")
    insights_path   = os.path.join(dataset_dir, "insights.json")
    fi_path         = os.path.join(dataset_dir, "feature_importance.json")

    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        df = pd.read_csv(clean_path)
    except Exception as e:
        logger.error(f"Dashboard config failed to load dependencies: {str(e)}")
        return False

    metrics_def = {}
    if os.path.exists(metrics_def_path):
        try:
            with open(metrics_def_path) as f:
                metrics_def = json.load(f)
        except Exception:
            pass

    insights = {}
    if os.path.exists(insights_path):
        try:
            with open(insights_path) as f:
                insights = json.load(f)
        except Exception:
            pass

    fi_data = {}
    if os.path.exists(fi_path):
        try:
            with open(fi_path) as f:
                fi_data = json.load(f)
        except Exception:
            pass

    date_col     = schema.get("date_column")
    sales_col    = schema.get("sales_column")
    product_col  = schema.get("product_column")
    region_col   = schema.get("region_column")
    customer_col = schema.get("customer_column")
    qty_col      = schema.get("quantity_column")
    profit_col   = schema.get("profit_column")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    def safe_agg(x_col, y_col, top_n=10, ascending=False):
        """Group by x_col, sum y_col, return top N records."""
        if not x_col or x_col not in df.columns:
            return []
        if not y_col or y_col not in df.columns:
            return []
        try:
            agg = (df.groupby(x_col)[y_col]
                     .sum()
                     .sort_values(ascending=ascending)
                     .head(top_n)
                     .reset_index())
            # Truncate long string labels
            if df[x_col].dtype == object:
                agg[x_col] = agg[x_col].astype(str).str[:20]
            return agg.to_dict(orient="records")
        except Exception:
            return []

    def safe_time_series(x_col, y_col, points=30):
        """Return time-sorted sample for a line/area chart."""
        if not x_col or x_col not in df.columns:
            return []
        if not y_col or y_col not in df.columns:
            return []
        try:
            tmp = df[[x_col, y_col]].dropna().copy()
            tmp[x_col] = pd.to_datetime(tmp[x_col], errors="coerce")
            tmp = tmp.dropna(subset=[x_col]).sort_values(x_col)
            # Monthly aggregation if > 30 rows
            if len(tmp) > points:
                tmp[x_col] = tmp[x_col].dt.to_period("M").astype(str)
                tmp = tmp.groupby(x_col)[y_col].sum().reset_index()
            else:
                tmp[x_col] = tmp[x_col].astype(str)
            return tmp.tail(points).to_dict(orient="records")
        except Exception:
            return []

    charts = []

    # ── Chart 1: Sales Trend Over Time (line) ────────────────────────────────
    if date_col and sales_col:
        data = safe_time_series(date_col, sales_col)
        if data:
            charts.append({
                "id": "trend_line",
                "type": "line",
                "title": f"{sales_col} Trend Over Time",
                "x": date_col,
                "y": sales_col,
                "data": data
            })

    # ── Chart 2: Sales by Product (bar) ─────────────────────────────────────
    if product_col and sales_col:
        data = safe_agg(product_col, sales_col)
        if data:
            charts.append({
                "id": "product_bar",
                "type": "bar",
                "title": f"Top Products by {sales_col}",
                "x": product_col,
                "y": sales_col,
                "data": data
            })

    # ── Chart 3: Sales by Region (pie) ──────────────────────────────────────
    if region_col and sales_col:
        data = safe_agg(region_col, sales_col)
        if data:
            charts.append({
                "id": "region_pie",
                "type": "pie",
                "title": f"{sales_col} by Region",
                "x": region_col,
                "y": sales_col,
                "data": data
            })

    # ── Chart 4: Revenue Growth Area (area) ─────────────────────────────────
    if date_col and sales_col:
        data = safe_time_series(date_col, sales_col)
        if data:
            charts.append({
                "id": "revenue_area",
                "type": "area",
                "title": "Cumulative Revenue Growth",
                "x": date_col,
                "y": sales_col,
                "data": data
            })

    # ── Chart 5: Top Customers by Sales (horizontal bar) ────────────────────
    if customer_col and sales_col:
        data = safe_agg(customer_col, sales_col, top_n=8)
        if data:
            charts.append({
                "id": "customer_bar",
                "type": "bar",
                "title": f"Top Customers by {sales_col}",
                "x": customer_col,
                "y": sales_col,
                "data": data,
                "horizontal": True
            })
    # Fallback: if no customer col, use any categorical
    elif sales_col:
        cat_candidates = [c for c in df.select_dtypes(include="object").columns
                          if c not in [product_col, region_col, date_col] and df[c].nunique() < 30]
        if cat_candidates:
            cat_col = cat_candidates[0]
            data = safe_agg(cat_col, sales_col, top_n=8)
            if data:
                charts.append({
                    "id": "customer_bar",
                    "type": "bar",
                    "title": f"Sales by {cat_col}",
                    "x": cat_col,
                    "y": sales_col,
                    "data": data,
                    "horizontal": True
                })

    # ── Chart 6: Quantity Sold per Product (bar) ─────────────────────────────
    if qty_col and product_col:
        data = safe_agg(product_col, qty_col)
        if data:
            charts.append({
                "id": "qty_bar",
                "type": "bar",
                "title": f"Quantity Sold by Product",
                "x": product_col,
                "y": qty_col,
                "data": data
            })
    # Fallback: quantity × best categorical
    elif qty_col:
        cat_col = next((c for c in df.select_dtypes(include="object").columns
                        if df[c].nunique() < 20), None)
        if cat_col:
            data = safe_agg(cat_col, qty_col)
            if data:
                charts.append({
                    "id": "qty_bar",
                    "type": "bar",
                    "title": f"Quantity by {cat_col}",
                    "x": cat_col,
                    "y": qty_col,
                    "data": data
                })

    # ── Chart 7: Feature Importance (bar) ───────────────────────────────────
    fi_importance = fi_data.get("importance", {})
    if fi_importance:
        top_fi = sorted(fi_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        fi_chart_data = [{"feature": k, "importance": round(v, 4)} for k, v in top_fi]
        charts.append({
            "id": "feature_importance_bar",
            "type": "bar",
            "title": "Feature Importance",
            "x": "feature",
            "y": "importance",
            "data": fi_chart_data
        })
    # Fallback chart 7: Profit trend if fi_data missing
    elif profit_col and date_col:
        data = safe_time_series(date_col, profit_col)
        if data:
            charts.append({
                "id": "profit_trend",
                "type": "area",
                "title": f"{profit_col} Trend",
                "x": date_col,
                "y": profit_col,
                "data": data
            })
    # Fallback chart 7b: Numeric distribution via variance bar
    elif len(numeric_cols) >= 2 and sales_col:
        second_numeric = next((c for c in numeric_cols if c != sales_col), None)
        if second_numeric:
            cat_col = next((c for c in df.select_dtypes(include="object").columns
                            if df[c].nunique() < 15), None)
            if cat_col:
                data = safe_agg(cat_col, second_numeric)
                if data:
                    charts.append({
                        "id": "secondary_metric_bar",
                        "type": "bar",
                        "title": f"{second_numeric} by {cat_col}",
                        "x": cat_col,
                        "y": second_numeric,
                        "data": data
                    })

    # ── Ensure minimum 6 charts with numeric fallbacks ───────────────────────
    if len(charts) < 6 and len(numeric_cols) >= 2 and sales_col in numeric_cols:
        for extra_col in numeric_cols:
            if extra_col == sales_col:
                continue
            cat_col = next((c for c in df.select_dtypes(include="object").columns
                            if df[c].nunique() < 20 and c != date_col), None)
            if cat_col:
                data = safe_agg(cat_col, extra_col)
                if data and len(charts) < 7:
                    charts.append({
                        "id": f"extra_{extra_col}",
                        "type": "bar",
                        "title": f"{extra_col} by {cat_col}",
                        "x": cat_col,
                        "y": extra_col,
                        "data": data
                    })
            if len(charts) >= 7:
                break

    config = {
        "charts":            charts,
        "insights":          insights.get("insights", []),
        "executive_summary": insights.get("summary", "")
    }

    out_path = os.path.join(dataset_dir, "dashboard_config.json")
    with FileLock(out_path + ".lock"):
        with open(out_path, "w") as f:
            json.dump(config, f, indent=4)

    logger.info(f"Dashboard Config generated with {len(charts)} charts.")
    return config


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = generate_dashboard_config(sys.argv[1])
        print(json.dumps({"charts_count": len(res.get("charts", []))}))
    else:
        print(json.dumps({"error": "Usage: python dashboard.py <dataset_dir>"}))
