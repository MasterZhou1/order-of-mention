"""Write every table of the paper as LaTeX into paper/tables/ from paper/data. No model is loaded.

    python paper/make_tables.py

Table 1  selection-evidence-table.tex   transfer.json
Table 2  rearranged-table.tex           transfer.json
Table 3  composition-table.tex          composition.json
Table 4  development-table.tex          olmo2_checkpoints.json
Table 5  generality-scale-table.tex     scale_sweep.json
Table 6  generality-base-table.tex      scale_sweep.json, invariance.json, query_geometry.json
Table 7  generality-prose-table.tex     prose.json

Rates are designated answers (%). A "+-x" entry is half the 95% word-set bootstrap interval; Tables 1 and 3 halve the
interval after rounding its endpoints to 0.1 points, Tables 2 and 7 halve the unrounded interval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import tables  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "tables"
    out.mkdir(exist_ok=True)
    for name, text in tables.build_all().items():
        (out / name).write_text(text)
        print(f"wrote {out / name}")
