"""Writes notebooks/classification_experiments.ipynb (run from Phase_3, then execute it).

    python notebooks/build_experiments_notebook.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Classification experiments: results\n\n"
       "Executed companion to [`docs/classification.md`](../docs/classification.md) (Part B), which explains every "
       "step. The notebook trains nothing: it reads the saved results in `outputs/classification/experiments/` "
       "(written by `src/classification/run_experiments.py`) and redraws the figures. The frozen v1 forest "
       "(test PR-AUC 0.375, recall in the top 20% 0.532) is the headline result."),
    code("import sys, json\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display, Markdown\n"
         "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
         "from classification.experiments import EXP_DIR, TABLES_DIR, FIGURES_DIR, V1_DIR\n"
         "from classification import experiment_figures\n"
         "pd.set_option('display.max_colwidth', 80); pd.set_option('display.precision', 4)\n"
         "t = lambda n: pd.read_csv(TABLES_DIR / f'{n}.csv')\n"
         "fig = lambda n: display(Image(filename=str(FIGURES_DIR / f'{n}.png')))\n"
         "experiment_figures.make_all()   # redraw every figure from the saved tables"),
    md("## 1. Train / validation / test split\n\nFive expanding-window folds validate on Jan–May 2018; each trains on "
       "orders with T from Jan 2017 up to the fold's cut-off minus a 27-day purge gap. The test window is Jun–Aug 2018."),
    code("fig('E01_split_timeline')\npd.read_csv(V1_DIR / 'tables' / 'fold_table.csv')"),
    md("## 2. Processing pipeline (fitted inside every fold)"),
    code("fig('E02_pipeline')"),
    md("## 3. Experiments and their effect\n\nEach row changes one thing against the step's reference; `decision` "
       "applies the agreed rule. Dots in the chart are the five monthly differences. The test columns were added after every decision was frozen and are a secondary reference only."),
    code("led = t('ledger')\nled[['step','experiment','mean_val_pr_auc','d_pr_auc','months_better','d_recall','d_roc_auc',"
         "'d_brier','d_pr_delivered','train_minus_val_pr_auc','decision','test_pr_auc','d_test_pr_auc']]"),
    code("fig('E03_ablation'); fig('E04_supporting_criteria')"),
    md("## 4. Class balancing\n\nEvery family at its tuned hyperparameters with no correction, class weights "
       "(balanced, 2:1), random undersampling, random oversampling and SMOTE. The reference is balanced weights; "
       "`passes_rule` = Brier lower in ≥ 4/5 months with PR-AUC not lower by more than 0.005. Test columns are a "
       "secondary evaluation."),
    code("cb = t('class_balance')\ncb[['type','family','strategy','train_pr_auc_cv','val_pr_auc','val_brier','val_mean_score',"
         "'val_recall_at_capacity','test_pr_auc','test_brier','test_recall_at_capacity','passes_rule','chosen_by_rule']]"),
    code("fig('E11_class_balance')"),
    md("## 5. Stacking\n\nOne convex stack of the tuned logistic, random-forest and HGB models, the counterpart "
       "of the regression Ridge/RF/HGB stack. In each fold the weights (non-negative, summing to one, no intercept, "
       "minimising log loss) are fitted on base-model scores for the last three months inside that fold's training "
       "rows; the validation month is never used to fit them. Kept only if it beats the best single model by the "
       "feature rule. The test row is a secondary evaluation scored after the decision was saved."),
    code("display(pd.Series(json.loads((EXP_DIR / 'stacking.json').read_text())))\n"
         "display(t('stacking_folds')); display(t('stacking_inner_audit')); fig('E12_stacking')"),
    md("## 6. Hyperparameter tuning\n\nRandom forest and HGB: two-stage screened search (stage 1 on folds 3–5, stage 2 "
       "the best three on all folds). Logistic: C grid. Decision tree: 20 random candidates. CatBoost: v1 parameters."),
    code("display(t('screened_search_random_forest').head(8)); display(t('screened_search_hist_gb').head(8))\n"
         "display(t('search_logistic')); fig('E07_screened_search')"),
    md("## 7. Train, each validation month, pooled validation and test, every family"),
    code("pf = t('per_fold'); best = led[led.experiment.str.startswith('best ') | (led.step=='S0')]\n"
         "pf[pf.key.isin(best.key)].pivot_table(index='experiment', columns='fold', values='val_pr_auc').round(4)"),
    code("fig('E05_validation_by_month'); t('test_results_v2')"),
    code("fig('E06_train_validation_test'); t('paired_bootstrap_v2_vs_v1')"),
    md("## 8. Orders delivered by T"),
    code("fig('E10_subgroups')"),
    md("## 9. Calibration and daily capacity"),
    code("display(t('calibration_forward')); display(t('calibration_decisions')); display(t('calibration_test_addendum'))\nfig('E08_reliability')"),
    code("t('daily_batch_recall')"),
    md("## 10. What the model relies on"),
    code("fig('E09_importance_v2'); t('importance_groups_v2')"),
    code("sel = json.loads((EXP_DIR / 'selection_v2.json').read_text())\n"
         "Markdown(f\"**v2 family:** {sel['v2_family']}  \\n**Recommended over v1 by the agreed rule:** {sel['recommended']}\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("classification_experiments.ipynb")
nbf.write(nb, out)
print("wrote", out)
