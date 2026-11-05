"""Assemble appendix.md for the final report from repository sources.

    python docs/final_report/assemble.py          (run from the repository root; then build.js, see README.md)
"""
import re
from pathlib import Path

import pandas as pd

F = Path("docs/final_report")
R, C = Path("outputs/regression/tables"), Path("outputs/classification")
PIPE, BREAK = "\\|", "\\pagebreak"


def md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = "" if pd.isna(v) else fmt.get(c, "{:.4f}").format(v)
            cells.append(str(v).replace("|", PIPE))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


A = ["# Appendix A. Data sources and cohorts", "",
     "::: Table A1. Olist source tables (Phase 1 data-quality audit)", "",
     "| Table | Rows | Rows per key | Missing values |", "|---|---|---|---|",
     "| orders | 99,441 | 1 per order | approval 0.16%, carrier 1.79%, delivery 2.98% |",
     "| order_items | 112,650 | 1.14 per order | none |",
     "| payments | 103,886 | 1.04 per order | none |",
     "| reviews | 99,224 | 1.01 per order | comment title 88.3%, comment text 58.7% |",
     "| customers | 99,441 | 1 per order | none |",
     "| sellers | 3,095 | 1 per seller | none |",
     "| products | 32,951 | 1 per product | category 1.85%, dimensions 0.01% |",
     "| geolocation | 1,000,163 | many per prefix | none (261,831 duplicate rows) |",
     "| category translation | 71 | 1 per category | none |", "",
     "::: Table A2. Regression cohort flow", "", md_table(pd.read_csv(R / "cohort_flow.csv")), "",
     "::: Table A3. Classification cohort flow", "",
     "| Step | Orders | Removed |", "|---|---|---|",
     "| All orders | 99,441 | |", "| At least one line item | 98,666 | 775 |", "| Has a review | 97,917 | 749 |",
     "| Review answered after T | 97,490 | 427 |", "| T in Jan 2017 – Aug 2018 | 97,187 | 303 |", "",
     "::: Table A4. Regression folds: training and scored orders, label coverage at selection", "",
     md_table(pd.read_csv(R / "fold_table.csv")[["fold", "cutoff", "train_n", "score_n", "window_completed_n",
                                                  "score_label_coverage", "pending_removed_after_gap"]],
              {"score_label_coverage": "{:.3f}"}), "",
     "::: Table A5. Classification folds under the v1 and strict rules", "",
     md_table(pd.read_csv(C / "strict_cutoff/tables/strict_fold_table.csv")[
         ["fold", "cutoff", "train_rows_v1", "train_rows_strict", "scored_rows_v1", "scored_rows_strict",
          "scored_low_review_rate_v1"]], {"scored_low_review_rate_v1": "{:.3f}"}), ""]

# Appendix B: the audited feature and model appendix, renumbered A -> B
src = Path("docs/report_appendix_features_models.md").read_text(encoding="utf-8")
src = re.sub(r"^# Appendix: (.*)$", lambda m: "# Appendix B. " + m.group(1)[0].upper() + m.group(1)[1:], src, flags=re.M)
src = re.sub(r"^## A(\d)\.", r"## B\1.", src, flags=re.M)
src = re.sub(r"\bA([1-9])\b", r"B\1", src)
src = re.sub(r"\[([^\]]+)\]\((?!http)[^)]+\)", r"\1", src)   # drop relative repository links

cl = Path("docs/classification.md").read_text(encoding="utf-8")
rc = Path("docs/regression/phase2_stacking_chronological.md").read_text(encoding="utf-8")
sh = Path("docs/regression/selected_stacking_holdout.md").read_text(encoding="utf-8")


def block(text, start, end):
    i = text.index(start)
    return text[i: text.index(end, i)].strip()


def drop_last_column(table):
    """Remove the last column of a Markdown table (used to drop Phase 2 random-split results)."""
    return "\n".join("|".join(row.rstrip().rstrip("|").split("|")[:-1]) + "|" for row in table.splitlines())


Cx = ["# Appendix C. Supplementary results", "",
      "::: Table C1. Regression: each family's selected candidate, validation MAE (mean of five months) and mean training MAE of the five fold fits", "",
      md_table(pd.read_csv(R / "cv_family_summary.csv").drop(columns="seconds")), "",
      "::: Table C2. Regression feature ablations on a fixed HGB (validation)", "",
      md_table(pd.read_csv(R / "diagnostic_summary.csv")), "",
      "::: Table C3. Regression test by month and by group (HGB_risk)", "",
      md_table(pd.read_csv(R / "test_by_month.csv")), "",
      md_table(pd.read_csv(R / "test_error_groups.csv")), "",
      "::: Table C4. Regression: all 14 Phase 2 combinations rerun chronologically (validation only)", "",
      drop_last_column(block(rc, "| Combination | Chronological", "SD describes")), "",
      "::: Table C5. Regression: selected ensembles, supplementary test assessment", "",
      block(sh, "| candidate | CV mean", "The table retains"), "",
      "::: Table C6. Classification: class-balancing study (each family at its tuned hyperparameters)", "",
      block(cl, "| Type | Family | Strategy |", "**What the class-balancing study shows**"), "",
      "::: Table C7. Classification Part B ledger (test columns are secondary)", "",
      block(cl, "| Step | Experiment |", "² Secondary").replace(" ²", ""), "",
      "::: Table C8. Classification: validation PR-AUC by month", "",
      block(cl, "| Type | Model | Jan |", "![train validation test]"), "",
      "::: Table C9. Classification strict cut-off rerun: effect of each rule (v1 forest fixed)", "",
      block(cl, "| Setting | Mean val PR-AUC |", "Scoring all validation orders"), "",
      "::: Table C10. Classification test by month (strict selection)", "",
      md_table(pd.read_csv(C / "strict_cutoff/tables/test_by_month.csv")), ""]
Cx = "\n".join(Cx).replace("**", "")

(F / "appendix.md").write_text("\n".join(A) + f"\n{BREAK}\n" + src + f"\n{BREAK}\n" + Cx, encoding="utf-8")
print("wrote", F / "appendix.md")
