"""Matched inputs and bounded model search; all primary targets use original days."""
import copy, time, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from . import config as C, data, audit

def specs():
    result={'OLS':{'family':'OLS','params':{}}}
    for a in [1.,10.,100.,1000.]:result[f'Ridge_a{a:g}']={'family':'Ridge','params':{'alpha':a}}
    for depth in [6,10]:
        for leaf in [30,100]:result[f'Tree_d{depth}_l{leaf}']={'family':'Tree','params':{'max_depth':depth,'min_samples_leaf':leaf}}
    for depth in [12,20,None]:
        for leaf in [30,80]:result[f'RF_d{depth}_l{leaf}']={'family':'RF','params':{'max_depth':depth,'min_samples_leaf':leaf,
            'n_estimators':100,'max_features':.7,'max_samples':.8}}
    for leaves in [15,31,63,127]:
        for l2 in [1.,10.]:result[f'HGB_leaves{leaves}_l2{l2:g}']={'family':'HGB','params':{
            'loss':'absolute_error','max_leaf_nodes':leaves,'l2_regularization':l2,'min_samples_leaf':30,
            'max_iter':300,'learning_rate':.05}}
    return result

def preprocessing(pack='calendar'):
    nums,cats=C.columns(pack)
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    categorical=Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
        ('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=30,max_categories=64,
                                drop='first',sparse_output=False))])
    return ColumnTransformer([('numeric',numeric,nums),('categorical',categorical,cats)],remainder='drop')

def estimator(spec):
    f=spec['family'];p=spec['params']
    if f=='OLS':return LinearRegression()
    if f=='Ridge':return Ridge(**p)
    if f=='Tree':return DecisionTreeRegressor(random_state=C.SEED,**p)
    if f=='RF':return RandomForestRegressor(random_state=C.SEED,n_jobs=4,**p)
    if f=='HGB':return HistGradientBoostingRegressor(random_state=C.SEED,early_stopping=False,**p)
    raise ValueError(f)

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert len(y)==len(p)>0 and np.isfinite(y).all() and np.isfinite(p).all()
    e=p-y
    return {'n':len(y),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),
        'r2':float(r2_score(y,p)) if len(y)>1 else None,'bias':float(np.mean(e)),
        'within3':float(np.mean(np.abs(e)<=3))}

def predict(model,X):
    p=model['learner'].predict(model['prep'].transform(X))
    if model.get('log_target'):p=np.expm1(p)
    return np.maximum(0,p)

def fit_model(frame,spec,pack='calendar',log_target=False):
    # Every entry point, including nested risk/ensemble paths, is behind this gate.
    audit.require_pass()
    prep=preprocessing(pack)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Found unknown categories.*')
        x=prep.fit_transform(frame)
    reg=estimator(spec)
    y=frame.lead_time_days.to_numpy()
    with threadpool_limits(limits=4): reg.fit(x,np.log1p(y) if log_target else y)
    return {'prep':prep,'learner':reg,'spec':copy.deepcopy(spec),'pack':pack,'log_target':log_target,
            'fit_n':len(frame),'fit_ids_sha256':data.ids_hash(frame.order_id),
            'input_columns':sum((list(v) for v in C.columns(pack)),[])}

def fit_on_arrays(x,y,spec):
    audit.require_pass();reg=estimator(spec)
    with threadpool_limits(limits=4):reg.fit(x,y)
    return reg

def preprocessor_audit(prep,frame):
    nums,cats=C.columns('calendar')
    med=frame[nums].median().fillna(0).to_numpy()
    np.testing.assert_allclose(prep.named_transformers_['numeric'].named_steps['impute'].statistics_,med)
    enc=prep.named_transformers_['categorical'].named_steps['encode']
    for name,values in zip(cats,enc.categories_):
        assert set(values)<=set(frame[name].dropna().unique())|{0}
    return {'fit_n':len(frame),'fit_ids_sha256':data.ids_hash(frame.order_id),
            'encoded_features':len(prep.get_feature_names_out()),'numeric_medians_verified':True,
            'categorical_dictionary_training_only':True}
