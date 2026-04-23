import pandas as pd
import os
import json
import logging
from filelock import FileLock
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger("system_logger")

def generate_forecast(dataset_dir, periods=30):
    """
    Implements basic time-series forecasting using ARIMA.
    Outputs the forecast.json artifact with confidence intervals.
    """
    cleaned_path = os.path.join(dataset_dir, "cleaned_data.csv")
    schema_path = os.path.join(dataset_dir, "schema.json")
    forecast_path = os.path.join(dataset_dir, "forecast.json")
    
    try:
        df = pd.read_csv(cleaned_path)
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data for forecasting: {str(e)}")
        return False
        
    date_col = schema.get("date_column")
    sales_col = schema.get("sales_column") # or a dynamically mapped revenue target
    
    # Safeguard missing required columns for forecasting
    if not date_col or not sales_col or date_col not in df.columns or sales_col not in df.columns:
        logger.info("Insufficient schema for forecasting (needs date and sales). Skipping.")
        with FileLock(forecast_path + ".lock"):
            with open(forecast_path, "w") as f:
                json.dump({"status": "skipped", "reason": "missing_datetime_or_sales_column"}, f, indent=4)
        return {"status": "skipped", "forecast_file": forecast_path}
        
    try:
        # Prepare Time Series data
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        ts_df = df.dropna(subset=[date_col, sales_col])
        ts_df = ts_df.groupby(date_col)[sales_col].sum().reset_index()
        ts_df = ts_df.set_index(date_col).sort_index()
        
        # Ensure daily frequency (fill missing with 0 or interpolate)
        ts_df = ts_df.asfreq('D').fillna(0)
        
        # Safeguard >=30 data points constraint
        if len(ts_df) < 30:
            logger.info(f"Not enough data points for reliable forecasting ({len(ts_df)} < 30). Skipping.")
            with FileLock(forecast_path + ".lock"):
                with open(forecast_path, "w") as f:
                    json.dump({"status": "skipped", "reason": "insufficient_time_points_min_30"}, f, indent=4)
            return {"status": "skipped", "forecast_file": forecast_path}
            
        # Fit ARIMA Model (using basic order for rapid execution without grid search)
        logger.info(f"Fitting ARIMA(5,1,0) for {sales_col}...")
        model = ARIMA(ts_df[sales_col].astype(float), order=(5, 1, 0))
        fitted = model.fit()
        
        # Forecast out 'periods' days
        forecast = fitted.get_forecast(steps=periods)
        predicted_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        # Format the output into JSON safely
        forecast_dates = predicted_mean.index.astype(str).tolist()
        predictions = predicted_mean.values.tolist()
        lower_bound = conf_int.iloc[:, 0].values.tolist()
        upper_bound = conf_int.iloc[:, 1].values.tolist()
        
        forecast_data = {
            "status": "success",
            "target": sales_col,
            "periods": periods,
            "dates": forecast_dates,
            "forecast": predictions,
            "confidence_intervals": {
                "lower": lower_bound,
                "upper": upper_bound
            }
        }
        
        with FileLock(forecast_path + ".lock"):
            with open(forecast_path, "w") as f:
                json.dump(forecast_data, f, indent=4)
            
        logger.info(f"Forecasting complete. Saved to {forecast_path}")
        return {"status": "success", "forecast_file": forecast_path}
        
    except Exception as e:
        logger.error(f"Forecasting failed to execute: {str(e)}")
        with FileLock(forecast_path + ".lock"):
            with open(forecast_path, "w") as f:
                json.dump({"status": "failed", "reason": str(e)}, f, indent=4)
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        result = generate_forecast(dataset_dir)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "Missing argument: dataset_dir"}))
