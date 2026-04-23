import pandas as pd
import numpy as np
import os
import json
import logging
from filelock import FileLock

logger = logging.getLogger("system_logger")

def extract_kpis(df, schema):
    """
    Computes static KPIs like Total Sales, Profit, AOV, Best/Worst Product, etc.
    """
    kpis = {}
    
    sales_col = schema.get("sales_column")
    profit_col = schema.get("profit_column")
    product_col = schema.get("product_column")
    region_col = schema.get("region_column")
    
    total_sales = 0
    if sales_col and sales_col in df.columns:
        total_sales = float(df[sales_col].sum())
        kpis["total_sales"] = total_sales
        kpis["average_order_value"] = float(df[sales_col].mean())
        
    if profit_col and profit_col in df.columns:
        total_profit = float(df[profit_col].sum())
        kpis["total_profit"] = total_profit
        if total_sales > 0:
            kpis["profit_margin_percentage"] = float((total_profit / total_sales) * 100)
            
    if product_col and sales_col and product_col in df.columns and sales_col in df.columns:
        prod_sales = df.groupby(product_col)[sales_col].sum()
        if len(prod_sales) > 0:
            kpis["best_product"] = {"name": str(prod_sales.idxmax()), "sales": float(prod_sales.max())}
            kpis["worst_product"] = {"name": str(prod_sales.idxmin()), "sales": float(prod_sales.min())}
            
    if region_col and sales_col and region_col in df.columns and sales_col in df.columns:
        reg_sales = df.groupby(region_col)[sales_col].sum()
        if len(reg_sales) > 0:
            kpis["best_region"] = {"name": str(reg_sales.idxmax()), "sales": float(reg_sales.max())}
            kpis["worst_region"] = {"name": str(reg_sales.idxmin()), "sales": float(reg_sales.min())}
            
    return kpis

def analyze_profitability(df, schema):
    """
    Computes explicit profitability analysis: margin per product, loss-making items.
    """
    insights = {}
    
    profit_col = schema.get("profit_column")
    product_col = schema.get("product_column")
    
    if profit_col and product_col and profit_col in df.columns and product_col in df.columns:
        prod_profit = df.groupby(product_col)[profit_col].sum().sort_values()
        
        # Loss making products
        losses = prod_profit[prod_profit < 0]
        insights["loss_making_products"] = [{"product": str(p), "loss": float(abs(l))} for p, l in losses.items()]
        
        # Highest profit products
        winners = prod_profit[prod_profit > 0].tail(5)
        insights["top_profitable_products"] = [{"product": str(p), "profit": float(l)} for p, l in winners.items()]
        
    return insights

def perform_root_cause_analysis(df, schema):
    """
    Statistical comparison to detect what drove changes (RCA).
    Simplified: Split by time midpoint and compare feature distributions.
    """
    rca = {}
    date_col = schema.get("date_column")
    sales_col = schema.get("sales_column")
    
    if date_col and sales_col and date_col in df.columns and sales_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(date_col)
            
            if len(df_sorted) >= 4:
                mid_idx = len(df_sorted) // 2
                period1 = df_sorted.iloc[:mid_idx]
                period2 = df_sorted.iloc[mid_idx:]
                
                sales1 = period1[sales_col].sum()
                sales2 = period2[sales_col].sum()
                
                delta_sales = sales2 - sales1
                delta_pct = (delta_sales / sales1) * 100 if sales1 > 0 else 0
                
                rca["period_comparison"] = {
                    "direction": "increased" if delta_sales > 0 else "decreased",
                    "percentage_change": round(delta_pct, 2)
                }
                
                # Check what feature changed the most (categorical distributions)
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                major_factors = []
                for col in cat_cols:
                    if col == date_col: continue
                    p1_counts = period1.groupby(col)[sales_col].sum()
                    p2_counts = period2.groupby(col)[sales_col].sum()
                    
                    diff = p2_counts - p1_counts
                    diff = diff.dropna()
                    
                    if len(diff) > 0:
                        biggest_dragger = diff.idxmin()
                        dragger_val = diff.min()
                        
                        biggest_driver = diff.idxmax()
                        driver_val = diff.max()
                        
                        if delta_sales < 0 and dragger_val < 0:
                            major_factors.append(f"{col} '{biggest_dragger}' sales decreased by {abs(dragger_val):.2f}")
                        elif delta_sales > 0 and driver_val > 0:
                            major_factors.append(f"{col} '{biggest_driver}' sales increased by {driver_val:.2f}")
                            
                rca["major_contributing_factors"] = major_factors[:5] # Top 5
        except Exception as e:
            logger.warning(f"RCA Failed: {str(e)}")
            
    return rca


def generate_recommendations(kpis, rca, prof):
    """
    Generates rule-based recommendations.
    """
    recs = []
    
    if "loss_making_products" in prof and len(prof["loss_making_products"]) > 0:
        worst = prof["loss_making_products"][0]["product"]
        recs.append(f"Review pricing or discontinue '{worst}' as it is currently generating the largest loss.")
        
    if "best_product" in kpis:
        best = kpis["best_product"]["name"]
        recs.append(f"Increase inventory and marketing spend for '{best}' to capitalize on high sales velocity.")
        
    if rca and "period_comparison" in rca:
        if rca["period_comparison"]["direction"] == "decreased":
            recs.append("Overall metric trend is downward. Implement targeted promotional discounts for dropping segments.")
            
    return recs

def run_bi_engine(dataset_dir):
    cleaned_path = os.path.join(dataset_dir, "cleaned_data.csv")
    schema_path = os.path.join(dataset_dir, "schema.json")
    
    try:
        df = pd.read_csv(cleaned_path)
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"BI Engine failed to load data: {str(e)}")
        return False

    kpis = extract_kpis(df, schema)
    prof = analyze_profitability(df, schema)
    rca = perform_root_cause_analysis(df, schema)
    recs = generate_recommendations(kpis, rca, prof)
    
    kpi_summary = {
        "kpis": kpis,
        "profitability_analysis": prof,
        "root_cause_analysis": rca,
        "recommendations": recs
    }
    
    out_path = os.path.join(dataset_dir, "kpi_summary.json")
    with FileLock(out_path + ".lock"):
        with open(out_path, "w") as f:
            json.dump(kpi_summary, f, indent=4)
        
    logger.info(f"BI Engine complete. Saved to {out_path}")
    return {"status": "success", "kpi_file": out_path}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_bi_engine(sys.argv[1])
