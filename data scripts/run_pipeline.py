import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "download_data.py",
    "clean_data.py",
    "outlier_analysis.py",
    "feature_preprocessing.py",
    "eda.py"
]


def main():
    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        print(f"\nRunning {script}...", flush=True)

        subprocess.run(
            [sys.executable, str(script_path)],
            check=True
        )

    print("\nEntire Phase 2 pipeline completed successfully!")


if __name__ == "__main__":
    main()