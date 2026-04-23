"""Generate a test dataset and run the full ML pipeline to verify everything works."""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add ml_engine to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(SCRIPT_DIR, "ml_engine")
sys.path.insert(0, ML_DIR)


def create_test_dataset():
    """Create a realistic sales dataset for testing."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Premium Z"]
    customers = [f"CUST-{i:04d}" for i in range(1, 51)]

    df = pd.DataFrame(
        {
            "date": dates,
            "region": np.random.choice(regions, n),
            "product": np.random.choice(products, n),
            "customer_id": np.random.choice(customers, n),
            "quantity": np.random.randint(1, 100, n),
            "unit_price": np.round(np.random.uniform(10, 500, n), 2),
            "revenue": np.round(np.random.uniform(100, 50000, n), 2),
            "profit": np.round(np.random.uniform(-500, 10000, n), 2),
        }
    )

    # Inject some nulls for cleaning
    df.loc[np.random.choice(n, 15), "revenue"] = np.nan
    df.loc[np.random.choice(n, 8), "region"] = np.nan
    df.loc[np.random.choice(n, 5), "unit_price"] = np.nan

    # Inject duplicates
    dup_rows = df.iloc[np.random.choice(n, 10)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def run_test():
    print("=" * 60)
    print("  DATAINSIGHTS.AI — FULL PIPELINE INTEGRATION TEST")
    print("=" * 60)

    # 1. Create test dataset
    test_dir = os.path.join(ML_DIR, "data", "users", "test_user", "test_dataset")
    os.makedirs(test_dir, exist_ok=True)

    test_csv = os.path.join(test_dir, "raw_data.csv")
    df = create_test_dataset()
    df.to_csv(test_csv, index=False)
    del df  # Release file handle
    print(f"\n[SETUP] Created test dataset: 510 rows, 8 cols")
    print(f"[SETUP] Saved to: {test_csv}")

    # 2. Run the pipeline
    from run_pipeline import run_pipeline

    print("\n[TEST] Running full ML pipeline...\n")
    pipeline_result = run_pipeline(
        test_csv, user_id="test_user", dataset_id_input="test_dataset"
    )

    print(f"\n[RESULT] Status: {pipeline_result['status']}")
    print(f"[RESULT] Pipeline Mode: {pipeline_result['pipeline_mode']}")
    print(f"[RESULT] Data Quality Score: {pipeline_result['data_quality_score']}")
    print(f"[RESULT] Artifacts: {pipeline_result['artifacts_generated']}")
    print(f"[RESULT] Error: {pipeline_result.get('error', 'None')}")

    # 3. Verify all artifacts exist
    print("\n[VERIFY] Checking artifacts...")
    expected = [
        "dataset_metadata.json",
        "schema.json",
        "profile_report.json",
        "metrics.json",
        "feature_importance.json",
        "kpi_summary.json",
        "forecast.json",
        "dashboard_config.json",
        "metrics_definition.json",
        "insights.json",
        "model_metrics.json",
        "cleaned_data.csv",
    ]

    all_ok = True
    for artifact in expected:
        fpath = os.path.join(test_dir, artifact)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {artifact} ({size} bytes)")
        if not exists:
            all_ok = False

    # 4. Verify dashboard config has charts
    dash_path = os.path.join(test_dir, "dashboard_config.json")
    if os.path.exists(dash_path):
        with open(dash_path) as f:
            dash = json.load(f)
        print(f"\n[DASHBOARD] Charts generated: {len(dash.get('charts', []))}")
        for chart in dash.get("charts", []):
            print(
                f"  - {chart['type']}: {chart['title']} ({len(chart.get('data', []))} data points)"
            )

    # 5. Test query engine
    print("\n[CHATBOT] Testing query engine...")
    from pipeline.query_engine import answer_question, DatasetArtifacts

    artifacts = DatasetArtifacts(test_dir)
    test_questions = [
        "What is the total revenue?",
        "Show me top 5 products",
        "What are the monthly trends?",
        "Give me a summary",
    ]

    chatbot_ok = True
    for q in test_questions:
        r = answer_question(q, artifacts)
        print(f"\n  Q: {q}")
        print(f"  A: {r['answer'][:100]}...")
        print(f"  Intent: {r['intent']} | Confidence: {r['confidence']}")
        if r["intent"] == "error" or r["confidence"] == 0:
            chatbot_ok = False

    print("\n" + "=" * 60)
    if all_ok and pipeline_result["status"] == "completed" and chatbot_ok:
        print("  ALL TESTS PASSED - PIPELINE IS FULLY FUNCTIONAL")
    else:
        print("  SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 60)

    return all_ok and pipeline_result["status"] == "completed"


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
