"""Reproduce the 31 October scoped leakage review without refitting models."""
from pathlib import Path
import sys,os,json,hashlib,zipfile,io
import numpy as np,pandas as pd,joblib
import argparse,subprocess
parser=argparse.ArgumentParser(description="Read-only regression leakage review; requires private caches from the completed audited run. No estimators are fitted.")
parser.add_argument('--archive',type=Path,required=True,help='Fixed course ZIP; checked against the recorded SHA256')
parser.add_argument('--private-root',type=Path,help='Existing regression private data/cache folder')
parser.add_argument('--models-root',type=Path,help='Existing final regression model folder')
parser.add_argument('--output',type=Path,required=True,help='New JSON review record; does not overwrite original audits or evaluation results')
args=parser.parse_args()
base=Path(__file__).resolve().parents[1]
scratch=base.parent
sys.path.insert(0,str(base/'src'))
from regression import config as C,data,splits,models,risk,score,audit
from regression_stacking import protocol as P
C.PRIVATE=args.private_root or base/'data/regression';C.OUT=base/'outputs/regression';C.TABLES=C.OUT/'tables';C.MODELS=args.models_root or base/'models/regression'
if args.output.resolve().is_relative_to(C.PRIVATE.resolve()):
 raise ValueError('Keep review records outside the original private cache folder.')
if args.output.exists():
 raise FileExistsError('Use a new output path; existing review evidence is retained.')
record={'scope':'read-only leakage check; no model fitting or project-file changes','checked_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()}
audit.require_pass();record['original_audit_source_and_data_match']=True
frozen=json.loads((C.OUT/'frozen_selection.json').read_text());changed=[]
for name,h in frozen['source_hashes'].items():
 if hashlib.sha256((scratch/'Phase_3/src/regression'/name).read_bytes()).hexdigest()!=h:changed.append(name)
record['sources_changed_since_freeze']=changed
assert set(changed)<= {'reporting.py','verify.py'}
df=data.load();assert len(df)==96203 and df.order_id.is_unique
archive=args.archive
os.environ['OLIST_ARCHIVE']=str(archive)
t,inv=data.read_tables();o=t['orders'];purchase=pd.to_datetime(o.order_purchase_timestamp);receipt=pd.to_datetime(o.order_delivered_customer_date)
y=(receipt-purchase).dt.total_seconds()/86400
valid=o.order_status.eq('delivered')&y.ge(0)&y.notna()&purchase.ge('2017-01-01')&purchase.lt('2018-09-01')
raw=pd.DataFrame({'order_id':o.loc[valid,'order_id'],'purchase_ts':purchase[valid],'delivered_ts':receipt[valid],'lead_time_days':y[valid]})
raw=raw.set_index('order_id').loc[df.order_id].reset_index()
for col in ['purchase_ts','delivered_ts','lead_time_days']:np.testing.assert_array_equal(raw[col],df[col])
record['raw_target_and_dates_recomputed']=len(raw)
# Independently evaluate boolean predicates, then compare the code's positional partition.
folds=splits.folds(df,39);final=splits.final(df,39);all_folds=folds+[final];out=[];test_ids=set(df.iloc[final.score].order_id)
for f in all_folds:
 mask=(raw.purchase_ts>=pd.Timestamp('2017-01-01'))&(raw.purchase_ts<f.cutoff-pd.Timedelta(days=39))&(raw.delivered_ts<f.cutoff)
 expected=np.flatnonzero(mask);np.testing.assert_array_equal(expected,f.train)
 sm=(raw.purchase_ts>=f.cutoff)&(raw.purchase_ts<f.end)
 if f.name!='final':sm&=raw.delivered_ts<pd.Timestamp('2018-06-01')
 np.testing.assert_array_equal(np.flatnonzero(sm),f.score)
 train=df.iloc[f.train];sc=df.iloc[f.score]
 assert not set(train.order_id)&set(sc.order_id)
 if f.name!='final':assert not (set(train.order_id)|set(sc.order_id))&test_ids
 pending=int(((raw.purchase_ts>=pd.Timestamp('2017-01-01'))&(raw.purchase_ts<f.cutoff-pd.Timedelta(days=39))&raw.delivered_ts.ge(f.cutoff)).sum())
 out.append({'fold':f.name,'train_n':len(train),'score_n':len(sc),'unavailable_training_labels':int(train.delivered_ts.ge(f.cutoff).sum()),'unavailable_selection_labels':int(sc.delivered_ts.ge('2018-06-01').sum()) if f.name!='final' else None,'pending_removed':pending})
record['independent_partitions']=out
# Check actual fitted preprocessing state, not merely the pipeline declaration.
for f in folds:
 v=joblib.load(C.PRIVATE/f'preprocessor_{f.name}_calendar.joblib');train=df.iloc[f.train]
 assert v['fit_hash']==data.ids_hash(train.order_id) and v['score_hash']==data.ids_hash(df.iloc[f.score].order_id)
 models.preprocessor_audit(v['prep'],train)
 # Outcome identifiers cannot be selected by the actual fitted transformer.
 used=[c for _,transformer,cs in v['prep'].transformers_ if not (isinstance(transformer,str) and transformer=='drop') and isinstance(cs,list) for c in cs]
 assert not set(used)&set(C.FORBIDDEN+['order_id','customer_unique_id','chronology_flag'])
record['actual_fold_preprocessors_checked']=5
# Verify feature construction is invariant to every outcome/status timestamp.
x=data.features(t)
changed_tables=dict(t);changed_tables['orders']=t['orders'].copy()
for col in ['order_delivered_customer_date','order_delivered_carrier_date','order_approved_at']:changed_tables['orders'][col]='2099-12-31 23:59:59'
changed_tables['orders']['order_status']='canceled'
mut=data.features(changed_tables);pd.testing.assert_frame_equal(x,mut,check_exact=True)
record['raw_outcome_perturbation_rows']=len(x)
# Prediction is invariant when the target and other after-purchase fields are deleted or poisoned.
bundle=joblib.load(C.MODELS/'final_model.joblib');sc=df.iloc[final.score].sample(160,random_state=731)
needed=[c for c in bundle['input_columns'] if c not in ['risk_p30','risk_unavailable']]
assert not set(needed)&set(C.FORBIDDEN+['order_id','customer_unique_id','chronology_flag'])
a=score.predict_bundle(sc,bundle);b=score.predict_bundle(sc[['order_id','purchase_ts',*needed]].copy(),bundle)
poison=sc.copy();poison['lead_time_days']=-999.;poison['delivered_ts']=pd.Timestamp('2099-01-01');poison['order_status']='poison';poison['review_score']=1;poison['order_approved_at']=pd.Timestamp('2099-01-01');poison['order_delivered_carrier_date']=pd.Timestamp('2099-01-01')
c=score.predict_bundle(poison,bundle);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
record['prediction_outcome_perturbation_sample_n']=len(sc)
# Reconstruct each cached risk fit's training membership, including the final classifier.
risk_windows=0;risk_partitions=0
for f,key in [(f,'outer_'+f.name) for f in folds]+[(final,'final_training')]:
 train=df.iloc[f.train];v=joblib.load(C.PRIVATE/f'risk_{key}.joblib');assert v['fit_hash']==data.ids_hash(train.order_id)
 assert train.delivered_ts.lt(f.cutoff).all()
 for r in v['audit']:
  date=pd.Timestamp(r['cutoff']);past=train[(train.purchase_ts<date)&(train.delivered_ts<date)]
  scored=train[train.purchase_ts.dt.strftime('%Y-%m').eq(r['month'])]
  assert len(past)==r['train_n'] and len(scored)==r['score_n']
  assert data.ids_hash(past.order_id)==r['fit_ids_sha256']
  assert not set(past.order_id)&set(scored.order_id)
  assert past.delivered_ts.lt(date).all() and scored.purchase_ts.ge(date).all()
  risk_windows+=1
 risk_partitions+=1
record['risk_partitions_reconstructed']=risk_partitions;record['risk_inner_windows_reconstructed']=risk_windows
# New all-14 suite: reconstruct outer and inner window ID hashes from raw-aligned orders.
dev=df[df.purchase_ts<pd.Timestamp('2018-06-01')].reset_index(drop=True)
timing=pd.read_csv(scratch/'Phase_3/outputs/regression_stacking_chronological/tables/time_audit.csv')
checked=0
for f in P.outer_folds(dev):
 tr=dev.iloc[f['train_idx']];sc=dev.iloc[f['score_idx']]
 rows=[('outer',f['cutoff'],tr,sc)]+[('inner',w['cutoff'],tr.iloc[w['train_idx']],tr.iloc[w['score_idx']]) for w in P.chronological_windows(tr,f['cutoff'])]
 for stage,cutoff,train,scored in rows:
  r=timing[(timing.outer==f['name'])&(timing.stage==stage)&pd.to_datetime(timing.cutoff).eq(cutoff)].iloc[0]
  assert P.ids_digest(train.order_id)==r.train_ids_sha256 and P.ids_digest(scored.order_id)==r.score_ids_sha256
  assert train.purchase_ts.lt(cutoff-pd.Timedelta(days=39)).all() and train.delivered_ts.lt(cutoff).all()
  assert scored.delivered_ts.lt(pd.Timestamp(r.meta_label_cutoff)).all()
  assert not set(train.order_id)&set(scored.order_id)
  checked+=1
record['new_14_suite_outer_inner_windows_reconstructed']=checked
# Supplementary final stack inner window identities.
tr=df.iloc[final.train]
hold=pd.read_csv(scratch/'Phase_3/outputs/regression_stacking_holdout/tables/meta_time_audit.csv')
for w in P.chronological_windows(tr,pd.Timestamp('2018-06-01')):
 a,b=tr.iloc[w['train_idx']],tr.iloc[w['score_idx']];r=hold[hold.month.eq(w['name'])].iloc[0]
 assert len(a)==r.train_n and len(b)==r.score_n
 assert P.ids_digest(a.order_id)==r.train_ids_sha256 and P.ids_digest(b.order_id)==r.score_ids_sha256
 assert a.delivered_ts.lt(w['cutoff']).all() and b.delivered_ts.lt('2018-06-01').all()
record['supplementary_stack_inner_windows_reconstructed']=len(hold)
record['passed']=True
record['review_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
record['source_sha256']={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src/regression','src/regression_stacking','src/common'] for p in sorted((base/folder).glob('*.py'))}
record['limitations']=['No hyperparameter search or model retraining repeated.', 'New all-14 stacking membership checked against raw-aligned orders and stored ID hashes; its private fitted matrices are not replayed by this review.', 'Raw purchase-feature invariance checked across 98,666 orders; final inference perturbation is a 160-order sample.', 'Historical snapshot availability, completed-delivery selection and prior test exposure remain limitations.']
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(record,indent=2,default=str)+'\n')
print(json.dumps(record,indent=2,default=str),flush=True)
