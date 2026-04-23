import sys

with open('err.txt', 'r', encoding='utf-16le', errors='replace') as f:
    err_text = f.read()
with open('err_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('err.txt content:\n')
    f.write(err_text)
    
with open('test_output.txt', 'r', encoding='utf-16le', errors='replace') as f:
    out_text = f.read()
with open('test_output_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('test_output.txt content:\n')
    f.write(out_text)
    
import subprocess
try:
    result = subprocess.run([sys.executable, 'run_pipeline.py', '--dataset_path', 'realistic_erp_sales.csv'], 
                           capture_output=True, text=True, check=False)
    with open('pipeline_run_stdout.txt', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    with open('pipeline_run_stderr.txt', 'w', encoding='utf-8') as f:
        f.write(result.stderr)
except Exception as e:
    with open('pipeline_run_stderr.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
