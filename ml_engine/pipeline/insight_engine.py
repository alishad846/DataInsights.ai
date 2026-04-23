import os
import json
import logging
import pandas as pd
import numpy as np
from filelock import FileLock
from scipy import stats

logger = logging.getLogger("system_logger")


class AIAnalystInsightEngine:
    """
    Self-Serve AI Analyst Mode.
    Reads artifacts and calculates statistical trends (Z-score, Slopes) emitting severity insights.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.kpi_path = os.path.join(dataset_dir, "kpi_summary.json")
        self.metric_path = os.path.join(dataset_dir, "metrics.json")
        self.schema_path = os.path.join(dataset_dir, "schema.json")
        self.data_path = os.path.join(dataset_dir, "cleaned_data.csv")
        self.insights_path = os.path.join(dataset_dir, "insights.json")

        self.insights = []

    def execute(self):
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            df = pd.read_csv(self.data_path)

            with open(self.metric_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load requisite files for Insight Engine: {e}")
            return False

        sales_col = schema.get("sales_column")
        date_col = schema.get("date_column")

        # 1. Z-Score Anomaly Detection
        if sales_col and sales_col in df.columns:
            z_scores = np.abs(stats.zscore(df[sales_col].fillna(0)))
            anomaly_count = len(np.where(z_scores > 3)[0])
            if anomaly_count > 0:
                self.insights.append(
                    {
                        "type": "anomaly_detection",
                        "description": f"Detected {anomaly_count} severe statistical anomalies in '{sales_col}' based on a 3-sigma Z-score threshold.",
                        "severity": "warning" if anomaly_count < 10 else "critical",
                    }
                )

        # 2. Percent Change & Trend Slope Detection
        if (
            date_col
            and sales_col
            and date_col in df.columns
            and sales_col in df.columns
        ):
            try:
                ts = df.copy()
                ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                ts = ts.dropna(subset=[date_col, sales_col])
                monthly = ts.set_index(date_col).resample("ME")[sales_col].sum()

                if len(monthly) >= 3:
                    y = monthly.values
                    x = np.arange(len(y))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    if slope > 0 and r_value > 0.6:
                        self.insights.append(
                            {
                                "type": "trend_slope",
                                "description": f"Strong positive linear acceleration detected in {sales_col} over time (R²={round(r_value**2, 2)}).",
                                "severity": "info",
                            }
                        )
                    elif slope < 0 and r_value < -0.6:
                        self.insights.append(
                            {
                                "type": "trend_slope",
                                "description": f"Severe declining demand detected in {sales_col} across the dataset timeline.",
                                "severity": "critical",
                            }
                        )

                    # Recent Drop (Percent Change threshold)
                    last = monthly.iloc[-1]
                    prev = monthly.iloc[-2]
                    if prev > 0:
                        pct_change = ((last - prev) / prev) * 100
                        if pct_change <= -15.0:
                            self.insights.append(
                                {
                                    "type": "declining_performance",
                                    "description": f"Recent {sales_col} dropped significantly by {abs(round(pct_change, 1))}% between the last two recorded months.",
                                    "severity": "warning"
                                    if pct_change > -30
                                    else "critical",
                                }
                            )
                        elif pct_change >= 20.0:
                            self.insights.append(
                                {
                                    "type": "accelerating_performance",
                                    "description": f"Recent {sales_col} spiked heavily by {round(pct_change, 1)}% in the most recent month.",
                                    "severity": "info",
                                }
                            )
            except Exception as e:
                logger.warning(f"Trend detection failed: {e}")

        # 3. Metric Checks
        if metrics.get("profit_margin"):
            margin = metrics["profit_margin"]
            if margin < 0.05 and margin > 0:
                self.insights.append(
                    {
                        "type": "profitability",
                        "description": f"Overall profit margin is critically thin at {round(margin * 100, 2)}%.",
                        "severity": "warning",
                    }
                )
            elif margin <= 0:
                self.insights.append(
                    {
                        "type": "profitability",
                        "description": "Dataset exhibits a net-negative profitability margin.",
                        "severity": "critical",
                    }
                )

        # Base Summary
        summary = f"Automated AI Analysis completed. Generated {len(self.insights)} dynamic statistical insights covering anomaly spikes, trend slopes, and multi-variable metric bounds."

        payload = {"summary": summary, "insights": self.insights}

        with FileLock(self.insights_path + ".lock"):
            with open(self.insights_path, "w") as f:
                json.dump(payload, f, indent=4)

        logger.info(f"Insight Engine generated {len(self.insights)} insights.")
        return True


def generate_insights(dataset_dir):
    engine = AIAnalystInsightEngine(dataset_dir)
    return engine.execute()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        generate_insights(sys.argv[1])
