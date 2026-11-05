"""Meaningful protocol tests on constructed boundary cases; no long training run."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from regression import splits,config as C
from regression.audit import boundary_check

class RegressionProtocol(unittest.TestCase):
    def test_label_at_cutoff_and_cross_month_validation(self):boundary_check()
    def test_gap_never_substitutes_for_label_maturity(self):
        frame=pd.DataFrame({'purchase_ts':pd.to_datetime(['2017-01-01','2017-01-02','2017-12-20']),
            'delivered_ts':pd.to_datetime(['2017-01-10','2018-02-01','2017-12-30'])})
        fold=splits.window(frame,'x','2018-01-01','2018-02-01',39)
        self.assertEqual(fold.train.tolist(),[0])
    def test_final_window_contains_all_observed_long_targets(self):
        frame=pd.DataFrame({'purchase_ts':pd.to_datetime(['2018-05-31','2018-06-01','2018-08-31','2018-09-01']),
            'delivered_ts':pd.to_datetime(['2018-06-10','2018-10-01','2018-10-20','2018-09-10'])})
        fold=splits.final(frame,39)
        self.assertEqual(fold.score.tolist(),[1,2])
    def test_main_schema_excludes_outcomes_and_absolute_dates(self):
        columns=sum((list(v) for v in C.columns()),[])
        self.assertFalse(set(columns)&set(C.FORBIDDEN))
        self.assertNotIn('purchase_year_month',columns)
        self.assertNotIn('purchase_elapsed_days',columns)
        self.assertEqual(len(columns),len(set(columns)))
    def test_gap_calibration_cannot_use_future_completion(self):
        f=pd.DataFrame({'purchase_ts':pd.to_datetime(['2017-01-01','2017-12-01']),
            'delivered_ts':pd.to_datetime(['2017-01-11','2018-03-01']),'lead_time_days':[10.,90.]})
        self.assertEqual(splits.gap_spec(f)['q99_days'],10)

if __name__=='__main__':unittest.main()
