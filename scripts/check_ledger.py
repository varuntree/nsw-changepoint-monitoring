"""Number check for this repository: every number the thesis quotes must be recomputed from results/.

The full number lint (scripts/check_numbers.py) reads the thesis LaTeX source, which is not in this repository.
This script runs the parts of it that need only the repository:

  1. the generated tables in tables/ and repo-tables/ equal a fresh run of scripts/thesis_tables.py (which runs
     scripts/tables.py and applies the wording of the final text);
  2. every row of the ledger (results/extra/quoted.csv, built by scripts/numbers_extra.py from ledger/*.csv and
     quoted_spec.csv: one row per number printed in the thesis, with the file it is printed in, the printed value
     and the result file, row and column it comes from) is recomputed from results/ and must equal the printed
     value; rows with the same id must agree unless a note explains why;
  3. every number in the captions of the generated tables passes the whitelist or a ledger row (the captions are
     read as scripts/tables.py writes them, the wording the ledger was written for; repo-tables/simconditions.tex
     and checks.tex are typed by hand and their numbers are quoted in the chapters, so they are not read here).

The ledger's file and anchor columns follow the chapter files as they were when the ledger was built. The final
wording of the thesis moved, cut or repeated some sentences but changed no number, so every ledger row still
recomputes; a few numbers that the final text prints in new places have no row of their own.

What it cannot check without the thesis source: that the prose of each chapter prints exactly the ledger's values
(the lint's token-by-token pass over the chapters). The token-by-token lint of a LaTeX source is
    python scripts/check_ledger.py --thesis /path/to/thesis      (a folder with chapters/ and tables/)
it finds each ledger row by the words around its number when the ledger was built, so on the final wording it
reports anchors that moved.
Usage: python scripts/check_ledger.py [--thesis DIR]   (make check-numbers; exit code 1 on any failure)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import check_numbers as C  # noqa: E402
import thesis_tables as TT  # noqa: E402


def fresh_thesis_tables(tex_dir=None):
    """Like check_numbers.fresh_tables, for the two folders written by thesis_tables.py."""
    rep, names = [], set()
    for name, (folder, text) in TT.build().items():
        names.add(name)
        q = folder / name
        if not q.exists() or q.read_text() != text:
            rep.append(dict(file=f"{folder.name}/{name}", line=0, token="", printed="", kind="table", context="",
                            reason="generated table is stale (differs from a fresh thesis_tables.py run)"))
        other = (TT.REPO_DIR if folder == TT.KEEP_DIR else TT.KEEP_DIR) / name
        if other.exists():
            rep.append(dict(file=f"{other.parent.name}/{name}", line=0, token="", printed="", kind="table", context="",
                            reason="table is in both tables/ and repo-tables/"))
    return names, rep


C.fresh_tables = fresh_thesis_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", type=Path, default=None)
    a = ap.parse_args()
    ledger = pd.read_csv(C.LEDGER)
    rules = C.load_whitelist()
    if a.thesis:
        chapters = sorted((a.thesis / "chapters").glob("*.tex"))
        tables = sorted((a.thesis / "tables").glob("*.tex"))
        rep, n_tok, n_q, n_w = C.run(chapters, tables, ledger, rules, tex_dir=a.thesis / "tables")
        mode = f"full lint against {a.thesis}"
    else:
        # 1. tables/ and repo-tables/ are fresh; 3. the captions are linted in the wording the ledger was written
        # for (the tables.py output, before thesis_tables.py renames terms; it changes no number)
        _, fresh = fresh_thesis_tables()
        with tempfile.TemporaryDirectory() as tmp:
            T = importlib.import_module("tables")
            old_out, T.OUT = T.OUT, Path(tmp)
            for f in T.ALL:
                f()
            T.OUT = old_out
            tables = sorted(Path(tmp).glob("*.tex"))
            rep, n_tok, n_q, n_w = C.run([], tables, ledger, rules, check_fresh=False)
        rep = pd.concat([pd.DataFrame(fresh, columns=rep.columns), rep], ignore_index=True)
        # a ledger row printed in a chapter cannot be found here; "dead" is checked only for rows whose every file
        # is a table in this repository
        here = {f"tables/{p.name}" for p in tables}
        files = ledger.set_index("id")["tex"].fillna("").to_dict()

        def checkable(row):
            fs = {x.split(":")[0].strip() for x in str(files.get(row["context"], "")).split(";") if x.strip()}
            return bool(fs) and fs <= here

        dead = rep["reason"].str.startswith("dead row")
        rep = rep[~dead | rep.apply(checkable, axis=1)]
        mode = "repository mode (no thesis source): tables fresh, ledger recomputed, table captions linted"
    n_rows = len(ledger)
    print(f"{mode}\n{n_rows} ledger rows recomputed from results/; {n_tok} tokens in the files read, {n_q} quoted, "
          f"{n_w} whitelisted; {len(rep)} failures")
    if len(rep):
        print(rep.groupby("reason").size().to_string())
        print(rep.head(20).to_string(index=False))
        sys.exit(1)
    print("check-numbers passed")


if __name__ == "__main__":
    main()
