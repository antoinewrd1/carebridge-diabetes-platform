"""Run the complete local analysis sequentially with bounded CPU concurrency."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from carebridge.config import ROOT

STAGES = [
    "ingest.build_db", "viz.descriptives", "viz.eda", "models.anomaly",
    "stats.distributions", "stats.tests", "stats.h1b_model", "models.readmission_models",
    "models.calibration", "models.thresholds", "models.shap_analysis",
    "models.leakage_sensitivity", "models.comorbidity", "models.fairness",
    "models.neural_phenotypes", "reports.final_report",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    args = parser.parse_args()
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "2"
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    for stage in STAGES[STAGES.index(args.from_stage):]:
        print(f"\n[run] carebridge.{stage}", flush=True)
        subprocess.run([sys.executable, "-u", "-m", f"carebridge.{stage}"],
                       cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
