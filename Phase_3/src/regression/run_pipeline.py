"""Run from Phase_3: python src/regression/run_pipeline.py --stage audit."""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',default='all')
    p.add_argument('--output-root',help='Write a completely separate fresh run outside the submitted evidence.')
    args=p.parse_args()
    from regression import config as C
    if args.output_root:
        root=Path(args.output_root).resolve()
        C.ROOT=root
        C.OUT=root/'outputs/regression';C.TABLES=C.OUT/'tables';C.FIGURES=C.OUT/'figures'
        C.PRIVATE=root/'data/regression';C.MODELS=root/'models/regression'
    if args.stage=='all':
        from regression import audit
        from regression.pipeline import run
        if not (C.OUT/'audit_pass.json').exists():audit.run()
        else:audit.require_pass()
        for stage in ['select','diagnostics','stacking','freeze','test','analysis','reference','figures','package']:
            if stage=='test' and (C.OUT/'test_summary.json').exists():continue
            run(stage)
        return
    if args.stage=='audit':
        from regression.audit import run
        run()
    else:
        from regression.pipeline import run
        run(args.stage)

if __name__=='__main__':main()
