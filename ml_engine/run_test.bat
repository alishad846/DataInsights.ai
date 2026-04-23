@echo off
.\venv\Scripts\python.exe run_pipeline.py --dataset_path realistic_erp_sales.csv > pipeline_output.txt 2>&1
echo Pipeline finished. Exit code: %errorlevel% > pipeline_status.txt
