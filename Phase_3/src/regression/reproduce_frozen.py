"""Rebuild the frozen selected pipeline from raw ZIP, without rerunning model selection."""
import argparse,os,json
from pathlib import Path
import numpy as np
import joblib
from . import config as C,data,splits,audit,models
from .finalise import fit_descriptor
from .score import predict_bundle

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--reference-output',default=str(C.OUT))
    args=parser.parse_args()
    reference=Path(args.reference_output).resolve()
    selection=json.loads((reference/'frozen_selection.json').read_text())
    expected=json.loads((reference/'test_summary.json').read_text())
    root=Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty output directory for an independent rebuild.')
    os.environ['OLIST_ARCHIVE']=str(Path(args.archive).resolve())
    os.environ.pop('OLIST_CSV_DIR',None)
    C.ROOT=root;C.OUT=root/'outputs/regression';C.TABLES=C.OUT/'tables';C.FIGURES=C.OUT/'figures'
    C.PRIVATE=root/'data/regression';C.MODELS=root/'models/regression'
    audit.run();df=data.load();f=splits.final(df,selection['gap_days'])
    train,test=df.iloc[f.train],df.iloc[f.score]
    if selection['deploy_ensemble']:raise NotImplementedError('This frozen rebuild entry is for the retained single pipeline.')
    bundle,work=fit_descriptor(train,C.TEST_START,selection['best_single'],'independent_final')
    prediction=predict_bundle(test,bundle)
    observed=models.metrics(test.lead_time_days,prediction)
    for metric in ['mae','rmse','r2','bias','within3']:
        np.testing.assert_allclose(observed[metric],expected['test'][metric],rtol=1e-10,atol=1e-10)
    joblib.dump(bundle,C.MODELS/'reproduced_model.joblib')
    record={'passed':True,'raw_archive_rebuilt':True,'new_empty_directory':True,'derived_caches_reused':False,
        'selected_regressor_and_temporal_auxiliary_pipeline_refitted':True,'full_search_repeated':False,
        'selected_candidate':selection['best_single']['candidate'],'test_n':len(test),'test_metrics':observed,
        'training_metrics':models.metrics(train.lead_time_days,models.predict(bundle,work)),
        'test_ids_sha256':data.ids_hash(test.order_id),'input_sha256':data.digest(C.PRIVATE/'orders.pkl')}
    data.save_json(C.OUT/'frozen_reproduction_check.json',record)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
