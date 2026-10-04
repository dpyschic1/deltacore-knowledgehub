import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "ingest.py",
    "embed.py",
    "custom_graphrag/extract.py",
    "custom_graphrag/entity_resolution.py",
    "custom_graphrag/build_graph.py",
]

def main():
    for script_name in PIPELINE:
        script_path = SCRIPT_DIR / script_name
        print(f"=== Running {script_name} ===")
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(f"!!! {script_name} failed with exit code {result.returncode}, stopping pipeline.")
            sys.exit(result.returncode)
    print("=== Pipeline complete ===")

main()
