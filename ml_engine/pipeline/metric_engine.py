import os
import json
import logging
import pandas as pd
import re
from filelock import FileLock

logger = logging.getLogger("system_logger")


class SemanticMetricEngine:
    """
    Centralized registry for business definitions and metrics.
    Translates schema columns into aggregated KPIs and resolves formula dependencies.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.schema_path = os.path.join(dataset_dir, "schema.json")
        self.data_path = os.path.join(dataset_dir, "cleaned_data.csv")
        self.def_path = os.path.join(dataset_dir, "metrics_definition.json")
        self.val_path = os.path.join(dataset_dir, "metrics.json")

        self.schema = {}
        self.df = None
        self.definitions = {}
        self.computed_values = {}

    def load_artifacts(self):
        if not os.path.exists(self.schema_path) or not os.path.exists(self.data_path):
            logger.error("Missing schema or data for metric engine")
            return False

        with open(self.schema_path, "r") as f:
            self.schema = json.load(f)

        self.df = pd.read_csv(self.data_path)
        return True

    def build_definitions(self):
        """
        Dynamically constructs metric definitions based on the available schema.
        """
        # Map canonical roles to actual DataFrame columns
        sales_col = self.schema.get("sales_column")
        profit_col = self.schema.get("profit_column")
        cost_col = self.schema.get("cost_column")
        qty_col = self.schema.get("quantity_column")
        date_col = self.schema.get("date_column")

        metrics = {}

        # Base Aggregations
        if sales_col and sales_col in self.df.columns:
            metrics["total_sales"] = {"column": sales_col, "aggregation": "sum"}
            metrics["average_sales"] = {"column": sales_col, "aggregation": "mean"}
            metrics["monthly_sales_growth"] = {
                "type": "time_series_growth",
                "column": sales_col,
            }

        if profit_col and profit_col in self.df.columns:
            metrics["total_profit"] = {"column": profit_col, "aggregation": "sum"}

        if cost_col and cost_col in self.df.columns:
            metrics["total_cost"] = {"column": cost_col, "aggregation": "sum"}

        if qty_col and qty_col in self.df.columns:
            metrics["total_quantity"] = {"column": qty_col, "aggregation": "sum"}

        # Formula-based derived metrics (Dependencies)
        # Revenue is synonymous with sales here, but if not defined:
        if "total_sales" in metrics:
            metrics["total_revenue"] = {"formula": "total_sales"}

        if "total_profit" in metrics and "total_revenue" in metrics:
            metrics["profit_margin"] = {"formula": "total_profit / total_revenue"}

        if (
            "total_revenue" in metrics and "total_cost" not in metrics
        ) and "total_profit" in metrics:
            metrics["inferred_total_cost"] = {"formula": "total_revenue - total_profit"}

        if "total_sales" in metrics and "total_quantity" in metrics:
            metrics["average_order_value"] = {"formula": "total_sales / total_quantity"}

        self.definitions = {"metrics": metrics}

    def _evaluate_base_metric(self, metric_name, config):
        """Evaluates a single base aggregation or time-series feature metric directly"""
        if "column" in config and "aggregation" in config:
            col = config["column"]
            agg = config["aggregation"]
            if col in self.df.columns:
                if agg == "sum":
                    return float(self.df[col].sum())
                if agg == "mean":
                    return float(self.df[col].mean())
                if agg == "max":
                    return float(self.df[col].max())
                if agg == "min":
                    return float(self.df[col].min())

        elif config.get("type") == "time_series_growth":
            # Just a placeholder calculation for growth
            date_col = self.schema.get("date_column")
            col = config.get("column")
            if date_col and col and date_col in self.df.columns:
                try:
                    ts = self.df.copy()
                    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                    ts = ts.dropna(subset=[date_col, col])
                    # Monthly resample sum to calculate basic % growth of last month vs prev
                    monthly = ts.set_index(date_col).resample("ME")[col].sum()
                    if len(monthly) >= 2:
                        last = monthly.iloc[-1]
                        prev = monthly.iloc[-2]
                        if prev != 0:
                            return float(((last - prev) / prev) * 100.0)
                except:
                    pass
        return None

    def resolve_metric(self, metric_name, visited=None):
        """
        Recursively revolves and computes a metric, handling formula dependencies graph safely.
        """
        if visited is None:
            visited = set()

        # Prevent circular logic
        if metric_name in visited:
            raise ValueError(f"Circular dependency detected in metric: {metric_name}")

        if metric_name in self.computed_values:
            return self.computed_values[metric_name]

        if metric_name not in self.definitions["metrics"]:
            return None

        config = self.definitions["metrics"][metric_name]
        visited.add(metric_name)

        if "formula" in config:
            formula = config["formula"]
            # Extract variable names from formula (words)
            dependencies = re.findall(r"[a-zA-Z_]+", formula)
            for dep in dependencies:
                # Recursively resolve dependencies
                dep_val = self.resolve_metric(dep, visited.copy())
                if dep_val is None:
                    return None
                # Replace the dependency in the formula string with its computed value
                formula = re.sub(rf"\b{dep}\b", str(dep_val), formula)
            try:
                # Safe math evaluation using ast instead of eval
                import ast
                import operator as op

                # Supported operators
                _safe_ops = {
                    ast.Add: op.add,
                    ast.Sub: op.sub,
                    ast.Mult: op.mul,
                    ast.Div: op.truediv,
                    ast.USub: op.neg,
                    ast.UAdd: op.pos,
                }

                def _safe_eval(node):
                    if isinstance(node, ast.Expression):
                        return _safe_eval(node.body)
                    elif isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        left = _safe_eval(node.left)
                        right = _safe_eval(node.right)
                        fn = _safe_ops.get(type(node.op))
                        if fn is None:
                            raise ValueError(
                                f"Unsupported operator: {type(node.op).__name__}"
                            )
                        return fn(left, right)
                    elif isinstance(node, ast.UnaryOp):
                        operand = _safe_eval(node.operand)
                        fn = _safe_ops.get(type(node.op))
                        if fn is None:
                            raise ValueError(
                                f"Unsupported unary operator: {type(node.op).__name__}"
                            )
                        return fn(operand)
                    else:
                        raise ValueError(
                            f"Unsupported expression: {type(node).__name__}"
                        )

                tree = ast.parse(formula, mode="eval")
                val = _safe_eval(tree)
                self.computed_values[metric_name] = round(float(val), 4)
                return self.computed_values[metric_name]
            except Exception as e:
                logger.error(
                    f"Formula evaluation failed for {metric_name} ({config['formula']}): {e}"
                )
                return None
        else:
            val = self._evaluate_base_metric(metric_name, config)
            if val is not None:
                self.computed_values[metric_name] = round(val, 4)
                return self.computed_values[metric_name]

        return None

    def execute(self):
        """Primary runner method. Emits definitions and computed values."""
        if not self.load_artifacts():
            return False

        self.build_definitions()

        for m in self.definitions["metrics"].keys():
            self.resolve_metric(m)

        # Write Definitions
        with FileLock(self.def_path + ".lock"):
            with open(self.def_path, "w") as f:
                json.dump(self.definitions, f, indent=4)

        # Write Computed Values
        with FileLock(self.val_path + ".lock"):
            with open(self.val_path, "w") as f:
                json.dump(self.computed_values, f, indent=4)

        logger.info(
            f"Metric Engine resolved {len(self.computed_values)} semantic metrics."
        )
        return True


def generate_metric_definitions(dataset_dir):
    """Interface matching the pipeline orchestration"""
    engine = SemanticMetricEngine(dataset_dir)
    return engine.execute()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = generate_metric_definitions(sys.argv[1])
        print(json.dumps({"status": "success"} if res else {"status": "failed"}))
