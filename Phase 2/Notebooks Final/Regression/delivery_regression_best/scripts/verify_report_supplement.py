"""Independently cross-check the public report's aggregate arithmetic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/report_supplement'


def main():
    wide=pd.read_csv(RESULTS/'tables/train_cv_test_comparison.csv')
    long=pd.read_csv(RESULTS/'tables/metrics_by_phase.csv')
    folds=pd.read_csv(RESULTS/'tables/cv_fold_metrics.csv')
    assert len(wide)==15 and wide.model.is_unique
    assert wide.Train_n.eq(64634).all() and wide.Test_n.eq(31836).all()
    assert len(folds)==75 and len(long)==45
    for row in wide.to_dict('records'):
        cv=folds.loc[folds.model.eq(row['model'])]
        assert len(cv)==5 and cv.n.sum()==64634
        for column,value in [('CV_MAE_mean_days',cv.MAE_days.mean()),
            ('CV_MAE_SD_sample_days',cv.MAE_days.std(ddof=1)),
            ('CV_MAE_SD_population_days',cv.MAE_days.std(ddof=0)),
            ('CV_RMSE_mean_days',cv.RMSE_days.mean()),('CV_R2_mean',cv.R2.mean())]:
            np.testing.assert_allclose(row[column],value,atol=1e-12,rtol=0)
        for phase,prefix in [('train','Train'),('test','Test')]:
            part=long.loc[long.model.eq(row['model']) & long.phase.eq(phase)].iloc[0]
            for metric in ['MAE_days','RMSE_days','R2']:
                np.testing.assert_allclose(row[f'{prefix}_{metric}'],part[metric],atol=1e-12,rtol=0)
        pooled=float(np.average(cv.MAE_days,weights=cv.n))
        np.testing.assert_allclose(row['OOF_pooled_MAE_days'],pooled,atol=1e-12,rtol=0)
    final=wide.loc[wide.model.eq('Final HGB (63 leaves)')].iloc[0]
    reference=json.loads((ROOT/'results/result_summary.json').read_text())
    np.testing.assert_allclose(final.Test_MAE_days,4.150491686642425,atol=1e-12,rtol=0)
    gains=pd.read_csv(RESULTS/'tables/hgb_improvement_vs_references.csv')
    np.testing.assert_allclose(gains.Final_HGB_MAE_reduction_days,
        gains.Test_MAE_days-final.Test_MAE_days,atol=1e-12,rtol=0)
    np.testing.assert_allclose(gains.Final_HGB_MAE_reduction_pct,
        100*(gains.Test_MAE_days-final.Test_MAE_days)/gains.Test_MAE_days,atol=1e-12,rtol=0)
    groups=pd.read_csv(RESULTS/'tables/hgb_test_error_groups.csv')
    for dimension,part in groups.groupby('dimension'):
        assert part.n.sum()==31836,dimension
        np.testing.assert_allclose(np.average(part.MAE_days,weights=part.n),final.Test_MAE_days,atol=1e-12,rtol=0)
    repeat=pd.read_csv(RESULTS/'tables/hgb_grouped_permutation_repeats.csv')
    importance=pd.read_csv(RESULTS/'tables/hgb_grouped_permutation_importance.csv')
    assert len(repeat)==35 and len(importance)==7
    for row in importance.to_dict('records'):
        part=repeat.loc[repeat.group.eq(row['group'])]
        assert len(part)==5 and part.sample_n.eq(8000).all()
        np.testing.assert_allclose(row['MAE_increase_mean_days'],part.MAE_increase_days.mean(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(row['MAE_increase_SD_days'],part.MAE_increase_days.std(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(part.MAE_increase_days,part.permuted_MAE_days-part.baseline_MAE_days,atol=1e-12,rtol=0)
    stack=pd.read_csv(RESULTS/'tables/stacking_outer_cv_comparison.csv')
    np.testing.assert_allclose(stack.loc[stack.model.eq('hgb'),'mean_fold_MAE_days'].iloc[0],
        final.CV_MAE_mean_days,atol=1e-12,rtol=0)
    protocol=json.loads((RESULTS/'protocol.json').read_text())
    assert protocol['interpretability']['risk_recomputed_after_permutation']
    assert protocol['interpretability']['refitting_or_selection_performed'] is False
    print('Verified: 15 Train–CV–Test rows, 75 folds, improvements, group totals and permutation arithmetic.')


if __name__=='__main__': main()
