"""Render every figure of the paper from paper/data into paper/figures/ (PDF and PNG).

    python paper/make_figures.py              # all figures
    python paper/make_figures.py setting      # only scripts whose name contains "setting"

Each script runs in its own process, so the styles of different figures never mix. No model is loaded.
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
# script -> figures it writes (paper figure number)
FIGURES = {
    "fig_order_cartoon.py": ["order-cartoon (Fig. 1a)"],
    "fig_access_teaser.py": ["access-teaser (Fig. 1b)"],
    "fig_setting.py": ["setting (Fig. 2a)"],
    "fig_relational_hypothesis.py": ["relational-hypothesis (Fig. 2b-d)"],
    "fig_query_organization.py": ["query-organization (Fig. 3)", "query-controls (Fig. 12)"],
    "fig_address_linear.py": ["address-linear (Fig. 4)"],
    "fig_selection_depth.py": ["selection-depth (Fig. 5)", "depth-geometry (Fig. 13)"],
    "fig_development.py": ["development (Fig. 6)"],
    "fig_within_word_set.py": ["within-world (Fig. 7)"],
    "fig_role_address.py": ["role-address (Fig. 8)"],
    "fig_relation_generality.py": ["relation-generality (Fig. 9)"],
    "fig_address_controls.py": ["address-controls (Fig. 10)"],
    "fig_longer_lists.py": ["longer-lists (Fig. 11)"],
}


def main(patterns):
    failed = []
    for script, outputs in FIGURES.items():
        if patterns and not any(p in script for p in patterns):
            continue
        t = time.time()
        run = subprocess.run([sys.executable, script], cwd=SCRIPTS, capture_output=True, text=True)
        status = "ok" if run.returncode == 0 else "FAILED"
        print(f"{status:6s} {script:30s} {time.time() - t:5.1f}s  -> {', '.join(outputs)}")
        warnings = [line for line in run.stdout.splitlines() if "layout warning" in line]
        for line in warnings:
            print("       " + line)
        if run.returncode:
            failed.append(script)
            print(run.stderr[-2000:])
    print(f"figures written to {SCRIPTS.parent / 'figures'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
