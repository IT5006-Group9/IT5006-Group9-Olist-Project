"""Source-backed figures, reading notebooks, report text and deployment handoff."""
from pathlib import Path
import json,sys,platform,subprocess,textwrap,hashlib
import numpy as np
import pandas as pd
import matplotlib
if 'IPython' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import date2num,DateFormatter
import nbformat
from . import config as C, data, splits, models, audit
from .pipeline import read,save_table,log

COLORS={'main':'#28547A','reference':'#B07829','alternate':'#7D7045','light':'#A5BDCE','ink':'#253340'}
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180})
def table(name):return pd.read_csv(C.TABLES/(name+'.csv'))
def draw(name):
    style()
    if name=='01_windows':
        folds=table('fold_table');fig,ax=plt.subplots(figsize=(10,4))
        start=pd.Timestamp('2017-01-01')
        for i,r in folds.iterrows():
            cutoff=pd.Timestamp(r.cutoff);end=pd.Timestamp(r.window_end_exclusive)
            stop=cutoff-pd.Timedelta(days=int(r.gap_days))
            ax.barh(i,date2num(stop)-date2num(start),left=date2num(start),height=.5,color=COLORS['main'])
            ax.barh(i,date2num(cutoff)-date2num(stop),left=date2num(stop),height=.5,color='#DFE6EB',hatch='///',edgecolor='#8F9DA5')
            ax.barh(i,date2num(end)-date2num(cutoff),left=date2num(cutoff),height=.5,color=COLORS['reference'])
        ax.set_yticks(range(len(folds)),['Jan validation','Feb validation','Mar validation','Apr validation','May validation','Jun-Aug test'])
        ax.xaxis_date();ax.xaxis.set_major_formatter(DateFormatter('%b %Y'));ax.invert_yaxis()
        ax.set_title('Expanding training windows with a 39-day embargo and observed labels')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=COLORS['main'],label='Purchase window for training'),
            Patch(facecolor='#DFE6EB',hatch='///',label='Embargo'),Patch(color=COLORS['reference'],label='Validation / test')],
            loc='upper left',bbox_to_anchor=(0,-.19),ncol=3,frameon=False)
        ax.set_xlabel('Purchase date; training also requires receipt before the cutoff')
    elif name=='02_model_comparison':
        t=table('train_cv_test_comparison');t=t[t.candidate!='Convex_stack'].sort_values('cv_mae')
        fig,ax=plt.subplots(figsize=(10,5));y=np.arange(len(t))
        for offset,col,label,color in [(-.22,'train_mae','Train',COLORS['light']),(0,'cv_mae','Validation mean',COLORS['main']),(.22,'test_mae','Test',COLORS['reference'])]:
            ax.barh(y+offset,t[col],height=.21,label=label,color=color)
        ax.set_yticks(y,t.candidate);ax.invert_yaxis();ax.set_xlabel('MAE (days)');ax.set_title('Frozen comparisons under the chronological protocol')
        ax.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0,-.13));ax.grid(axis='x',alpha=.15)
    elif name=='03_test_months':
        t=table('test_by_month');fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        for ax,col,label in [(axes[0],'mae','MAE (days)'),(axes[1],'bias','Mean predicted minus actual (days)')]:
            ax.bar(t.month,t[col],color=COLORS['main']);ax.axhline(0,color=COLORS['ink'],lw=.7);ax.set_ylabel(label)
            for i,r in t.iterrows():ax.annotate(f'{r[col]:.2f}',(i,r[col]),xytext=(0,5),textcoords='offset points',ha='center')
        fig.suptitle('Final model: monthly performance on June-August purchases')
    elif name=='04_tail_errors':
        t=table('test_error_groups').iloc[:4];fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(t.group,t.mae,color=COLORS['main']);axes[0].set_ylabel('MAE (days)')
        x=np.arange(4);axes[1].bar(x-.18,t.order_share*100,.36,label='Orders',color=COLORS['main'])
        axes[1].bar(x+.18,t.absolute_error_share*100,.36,label='Absolute error',color=COLORS['reference'])
        axes[1].set_xticks(x,t.group);axes[1].set_ylabel('Share (%)');axes[1].legend(frameon=False)
        for ax in axes:ax.tick_params(axis='x',labelrotation=0)
        fig.suptitle('Long-tail error magnitude and contribution are different quantities')
    elif name=='05_importance':
        t=table('permutation_importance_groups').sort_values('mean');fig,ax=plt.subplots(figsize=(8,4))
        ax.barh(t.group,t['mean'],xerr=t['std'],color=COLORS['main'],capsize=3);ax.axvline(0,color=COLORS['ink'],lw=.8)
        ax.set_xlabel('Increase in MAE after joint permutation (days)')
        ax.set_title('Feature reliance on observed May validation outcomes')
        ax.text(0,-.24,'Up to 3,000 orders; five repeats; bars show repeat SD, not confidence intervals.',transform=ax.transAxes,fontsize=9)
    elif name=='06_calendar_ablation':
        t=table('diagnostic_summary');t=t[t.candidate.isin(['HGB_no_calendar','HGB_calendar','HGB_absolute','HGB_risk'])]
        fig,ax=plt.subplots(figsize=(8,4));labels={'HGB_no_calendar':'No calendar','HGB_calendar':'Cyclical calendar',
            'HGB_absolute':'Calendar + absolute date','HGB_risk':'Calendar + past-only risk'}
        ax.barh([labels[x] for x in t.candidate],t.cv_mae,xerr=t.cv_sd,color=COLORS['main'],capsize=3)
        ax.set_xlabel('Mean validation MAE (days); error bars: monthly SD')
        ax.set_title('Matched HGB feature ablation: fixed model parameters')
    elif name=='07_random_reference':
        t=table('chronological_vs_random');fig,ax=plt.subplots(figsize=(7,4))
        bars=ax.bar(['Chronological main','Random reference'],t.mae,color=[COLORS['main'],COLORS['reference']])
        for b in bars:ax.annotate(f'{b.get_height():.3f}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,6),textcoords='offset points',ha='center')
        ax.set_ylabel('Test MAE (days)');ax.set_title('Fixed configuration under two evaluation procedures')
        ax.text(0,-.22,'Test populations and training eligibility differ; this is not an isolated algorithm gain.',transform=ax.transAxes,fontsize=9)
    elif name=='08_stack_weights':
        t=table('stacking_folds');fig,ax=plt.subplots(figsize=(8,3.8));bottom=np.zeros(len(t))
        for col,label,color in [('ridge_weight','Ridge',COLORS['main']),('rf_weight','RF',COLORS['reference']),('hgb_weight','HGB',COLORS['alternate'])]:
            ax.bar(t.fold,t[col],bottom=bottom,color=color,label=label);bottom+=t[col]
        ax.set_ylim(0,1);ax.set_ylabel('Convex weight');ax.set_title('One temporal stacking comparison');ax.legend(ncol=3,frameon=False,loc='upper left',bbox_to_anchor=(0,-.13))
    elif name=='09_label_coverage':
        t=table('fold_table');t=t[t.fold!='final'];fig,ax=plt.subplots(figsize=(8,3.8))
        ax.bar(['Jan','Feb','Mar','Apr','May'],t.score_label_coverage*100,color=COLORS['main'])
        ax.set_ylim(0,110);ax.set_ylabel('Completed outcomes available by 1 June (%)')
        for i,r in t.reset_index(drop=True).iterrows():ax.text(i,r.score_label_coverage*100+2,f'{r.score_label_coverage:.1%}',ha='center')
        ax.set_title('Validation-label coverage at final model selection')
    else:raise ValueError(name)
    fig.tight_layout();return fig

FIGURES=['01_windows','02_model_comparison','03_test_months','04_tail_errors','05_importance',
         '06_calendar_ablation','07_random_reference','08_stack_weights','09_label_coverage']
def figures():
    for name in FIGURES:
        fig=draw(name);fig.savefig(C.FIGURES/(name+'.png'),bbox_inches='tight');plt.close(fig)
    log('Nine source-backed figures generated.')

def markdown_table(frame):
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|',' / ')
    header='| '+' | '.join(map(str,frame.columns))+' |'
    return '\n'.join([header,'| '+' | '.join(['---']*len(frame.columns))+' |']+
                      ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
def package():
    audit.require_pass();df=data.load();selection=read('frozen_selection.json');summary=read('test_summary.json')
    C.directories();docs=C.ROOT/'docs/regression';notebooks=C.ROOT/'notebooks/regression'
    source=Path(__file__).resolve().parent
    if (C.ROOT/'src/regression').resolve()!=source:
        import shutil
        for name in ['regression','common']:
            shutil.copytree(source.parent/name,C.ROOT/'src'/name,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__'))
    docs.mkdir(parents=True,exist_ok=True);notebooks.mkdir(parents=True,exist_ok=True)
    primary=summary['test'];cv=table('train_cv_test_comparison');g=read('purge_gap.json');flow=table('cohort_flow')
    winner=summary['selected_candidate'];single=selection['best_single']['candidate'];stack=read('stacking_complete.json')
    catalogue()
    final=splits.final(df,selection['gap_days']);folds=splits.folds(df,selection['gap_days'])
    population=[]
    for label,idx in [('Final training',final.train),('Selection outcomes',np.concatenate([f.score for f in folds])),
                      ('Eventual validation outcomes',np.concatenate([f.all_score for f in folds])),('Chronological test',final.score)]:
        y=df.iloc[idx].lead_time_days
        population.append({'population':label,'n':len(y),'mean_days':y.mean(),'median_days':y.median(),
            'p90_days':y.quantile(.9),'over30_share':y.gt(30).mean(),'over60_share':y.gt(60).mean()})
    save_table(pd.DataFrame(population),'target_distribution_by_stage')
    selected_label='HGB with a past-only slow-delivery risk feature' if winner=='HGB_risk' else winner
    sensitivity=table('selection_maturity_sensitivity').set_index('candidate')
    risk_row=sensitivity.loc['HGB_risk'];ridge_row=sensitivity.loc[selection['family_winners']['Ridge']]
    validation_margin=ridge_row.five_month_mae-risk_row.five_month_mae
    test_scores=table('train_test_results');test_scores=test_scores[test_scores.split=='test'].set_index('candidate')
    plain_hgb_test=test_scores.loc[selection['family_winners']['HGB'],'mae']
    risk_test=test_scores.loc['HGB_risk','mae']
    groups=table('test_error_groups');slow=groups.iloc[2:4]
    slow_n=int(slow.n.sum());slow_share=slow.order_share.sum();slow_error_share=slow.absolute_error_share.sum()
    points=f'''# Regression report evidence guide — Phase 3

## Scope and main result

The task remains purchase-time estimation of fractional purchase-to-receipt days for observed completed deliveries. The fixed course archive and screened-median geography are retained. No new dataset or modelling family was introduced. The chronological procedure selected **{winner}**. Its test MAE was **{primary['mae']:.4f} days**, RMSE **{primary['rmse']:.4f} days**, R² **{primary['r2']:.4f}**, and ±3-day coverage **{primary['within3']:.2%}** on **{summary['test_n']:,}** June–August orders. These are retrospective historical results; this dataset was exposed during prior development.

## Report §6 — Split and validation design

Orders are assigned by purchase timestamp. Five expanding windows validate January–May 2018, starting training in January 2017. The final model is trained as of 1 June and evaluated on June–August purchases. A fixed **{g['q99_days']}-day purge gap** was calibrated from 2017 purchases already delivered before 1 January 2018 (Q99); every training label must also be observed before its fitting cutoff. A finite gap alone does not guarantee label availability. Q95={g['q95_days']} and no-gap checks retain this strict label rule.

Parameter selection uses only validation outcomes observed before 1 June. May coverage is 65.3%, so this selection set underrepresents late-May and slow deliveries. Report `fold_table.csv`, `selection_maturity_sensitivity.csv` and the post-selection `validation_outcome_maturity_diagnostic.csv` together. The latter uses eventual outcomes only to quantify the limitation, never to choose a new model. Completed-delivery coverage and an unlogged extraction cutoff remain limitations; this is not an unconditional forecast for unresolved/cancelled orders.

For the frozen risk-HGB, May MAE is 2.8003 on outcomes available at selection versus 4.4222 on all eventual completed May outcomes. This diagnostic does not revise the choice. Strict time availability prevents using future labels, but it does not remove maturity-selection bias.

## Report §7 — Models, features and tuning

OLS, Ridge, one decision tree, Random Forest and HGB share the same purchase-time feature schema, imputation, category pooling/encoding and scaling. Primary comparisons use the original target scale. Published `search_design.json` enumerates every configuration; `cv_all_candidates_folds.csv` records each fit and training/validation scores. Model differences are evaluated at matched inputs. Feature changes are a separate fixed-HGB ablation: no calendar, cyclical calendar, absolute dates and a past-only slow-order risk feature. RF log-target is a separate fixed-configuration sensitivity.

OLS, Ridge, the tree and RF fit squared-error objectives; HGB fits absolute-error loss. The comparison therefore evaluates learner-and-loss designs at matched inputs, rather than identifying a pure algorithm effect under identical optimisation objectives.

The risk feature is trained within each outer fitting partition using earlier monthly blocks and labels known before those blocks. January–March 2017 are an explicit cold-start period with a missing-risk flag; random OOF probabilities from Phase 2 are not reused. Auxiliary fitting does not add new raw information.

The frozen selected specification is `{json.dumps(selection['best_single']['spec'],sort_keys=True)}` with feature pack `{selection['best_single']['pack']}`. For the retained HGB, this means absolute-error loss, 300 iterations, learning rate 0.05, 15 maximum leaves, minimum leaf size 30 and L2 regularisation 1; early stopping is disabled. The auxiliary risk classifier uses the fixed settings in `diagnostics_design.json`, independently of the low-review classification task. All point predictions are constrained to nonnegative days.

One convex Ridge/RF/HGB stack was assessed with three inner chronological windows inside each outer training partition. Base hyperparameters were selected in the main CV, so this is not fully nested hyperparameter selection. Stack mean validation MAE={stack['cv_mae']:.4f}; relative gain versus the best single model={stack['relative_gain']:.2%}; retained={stack['retained']} under the predeclared 1% minimum-gain rule. If discarded, no final stack test score is invented.

## Report §8 — Results and interpretation

{markdown_table(cv)}

The risk-HGB has only a {validation_margin:.4f}-day mean-validation advantage over Ridge. Its rank changes from {int(risk_row.five_month_rank)} over five selection months to {int(risk_row.jan_apr_rank)} when May is removed, whereas Ridge ranks {int(ridge_row.jan_apr_rank)} over January–April. The feature-variant table and maturity-coverage table therefore qualify this selection; no claim of a stable or significant superiority is made. The no-risk HGB's test MAE is {plain_hgb_test:.4f} versus {risk_test:.4f} for the risk variant. This is a descriptive comparison: the selection was frozen before test assessment and was not switched to the lowest test row.

Train MAE is in-sample; CV is the unweighted mean of five monthly MAEs with sample SD (ddof=1); Test is one fixed June–August window. Blank stack Train/Test cells mean no final stack was fitted because it was discarded. Full-search CV is development evidence, not an unbiased nested estimate. References use only the matching training window's mean/median; promise predicts the quoted duration without fitting. No independent random validation subset is added.

`model_metrics_by_stage.csv` provides MAE, RMSE, R², signed bias and ±3-day coverage by stage; `target_distribution_by_stage.csv` explains changing outcome mix. In the chronological test, {slow_n:,} orders exceed 30 days ({slow_share:.2%} of orders) and contribute {slow_error_share:.2%} of total absolute error. Their individual errors remain large, but they do not account for most aggregate MAE. These distribution changes prevent interpreting {primary['mae']:.4f} versus the old random 4.1505 as isolated model improvement.

Report `test_by_month.csv`, `test_error_groups.csv` and `test_by_route.csv`. Group MAE and share of total absolute error answer different questions; the duration groups form a partition, while geography/chronology slices overlap. Permutation importance is measured on up to 3,000 May validation outcomes observed by June 1, with five joint group shuffles; repeat SD is not a confidence interval and importance is not causal. If risk is used, it is recomputed after shuffling its source inputs.

## Success criteria

The target set before new model fits was at least 5% lower test MAE than the training-median reference, and lower MAE than delivery promise. Observed reduction versus median: **{summary['success']['mae_reduction_vs_median']:.2%}**; median criterion met: **{summary['success']['median_criterion_met']}**; promise criterion met: **{summary['success']['promise_criterion_met']}**. These are project assumptions, not validated Olist service-level requirements. Meeting them does not establish ROI or delivery guarantees.

## Supporting figures

`outputs/regression/figures/01–09`: time windows; model comparison; test months; tail errors; importance; feature ablation; random reference; stacking weights; label coverage. All tables are in the adjacent `tables/` directory.

## Deployment and reproducibility

See `deployment_handoff.md`. The deployed model must be the frozen bundle, not the previous random-split model. App examples use held-out historical orders; this simulates new orders without a live feed. Keep actual outcomes for after-prediction comparison only. Report and slides must quote the same tables and model identifier.

## Limitations to retain

Historical test exposure; only one substantial earlier seasonal cycle; changing month/route mix; validation-label maturity selection; conditional completed-delivery population; archived metadata availability assumptions; residual long-tail error; bounded search; base-configuration selection before stack evaluation; prediction rather than causal intervention evidence. Random reference changes training eligibility and test populations, so its score difference is not pure algorithm improvement.

## References and AI declaration

Use the existing Phase 1/2 references for geography, temporal effects and long-tail motivation. Zaghloul et al. supports comparability of the old random design, not future-period validity. This implementation uses scikit-learn estimators and SciPy convex optimisation. Codex assisted with implementation, protocol checks, figures and documentation. The team remains responsible for the submitted work; do not imply the AI performed an independent external validation.
'''
    (docs/'report_evidence_guide.md').write_text(points)
    (docs/'decision_log.md').write_text(f'''# Regression decision log

| Decision | Rationale and evidence |
|---|---|
| Chronological main evaluation | TA email requires it; `protocol.json` |
| Jan–May expanding validation; Jun–Aug test | Shared calendar breakpoints with classification; regression uses purchase time |
| {g['q99_days']}-day gap plus strict known labels | Calibrated before first window; `purge_gap.json`, zero future labels in `fold_table.csv` |
| Selection labels known before Jun 1 | Avoid using May's future outcomes to select the June model; coverage and ranking sensitivity published |
| Matched feature comparison | Separate algorithm and feature effects; `search_design.json` |
| Explicit bounded search | Five monthly scores for each enumerated configuration, no test-guided expansion |
| Calendar and risk ablation | Address period-specific reliance and earlier random cross-fitting |
| Final single model: {single} | Minimum pre-test validation MAE; `diagnostics_complete.json` |
| Stack retained: {stack['retained']} | Require ≥1% validation MAE reduction; `stacking_complete.json` |
| Test frozen before assessment | Configuration and ID hashes; `frozen_selection.json`, `test_lock.json` |
| Random result reference only | Same chronological single-model specification, no selection; different populations acknowledged |
| Phase 2 kept unchanged | Separate Phase_3 implementation and output directories |
''')
    (docs/'presentation_notes.md').write_text(f'''# Regression presentation segment — approximately 1 minute 45 seconds

## Slide 1: Evaluation design and model ladder (about 55 seconds)

Use `01_windows.png`, then the compact model-comparison table or `02_model_comparison.png`.

“We predict purchase-to-receipt duration using information available at checkout. In response to feedback, we moved the main evaluation to five expanding monthly validation windows and a June–August test. We require labels to be known at each fitting cutoff; a {g['q99_days']}-day embargo is an additional guard. OLS, Ridge and tree models use matched inputs, and features are tested separately. Validation selected {selected_label}; the final historical test MAE is {primary['mae']:.2f} days.”

## Slide 2: Meaning, limitations and application (about 50 seconds)

Use `04_tail_errors.png` and, if room, `05_importance.png`.

“We report monthly and long-duration errors rather than only one average. An average error does not guarantee an individual arrival date. Estimates can support provisional customer communication and route review; unusually slow orders still need human follow-up. May's available labels cover only 65.3% of completed validation orders at selection time, and the historical test was previously exposed. We therefore treat this as a constrained retrospective assessment, not proof of future operational accuracy. The app uses the same frozen prediction bundle.”

## Q&A evidence

- Why time split? TA requirement and intended future-order use.
- Why both embargo and observed-label checks? No finite percentile gap guarantees every slow order has completed.
- Why lower random error? Different time mixture, eligibility and test population; not a better deployment guarantee.
- Why calendar ablation? Specific year-month patterns may not transfer to unseen months.
- Why no/yes stack? Predeclared ≥1% validation gain; actual result is in `stacking_complete.json`.
- What counts as success? ≥5% MAE improvement over training median plus beating promise; business impact still needs a prospective pilot.

Do not reuse Phase 2's 4.1505 days as the Phase 3 result. Replace numeric claims only from the frozen evidence tables, never from a later exploratory rerun.
''')
    (docs/'deployment_handoff.md').write_text('''# Regression deployment handoff

## Interface

```python
import sys
sys.path.insert(0, "src")
import pandas as pd
from regression.score import load_bundle, predict_lead_time
bundle = load_bundle()  # models/regression/final_model.joblib
orders = pd.read_pickle("data/regression/deployment_orders.pkl")
predictions = predict_lead_time(orders, bundle)
```

The input is a pandas DataFrame of prepared **purchase-time** order features. Exact column names, groups and availability assumptions are in `outputs/regression/tables/feature_catalogue.csv`; the bundle's `input_columns` is authoritative for the chosen configuration. `risk_p30` and its cold-start flag are derived internally, not supplied by the app. Source preparation and date features are defined in `regression.data.features`, not reconstructed ad hoc by the app.

Output: order_id, predicted_days, quoted_days, predicted_exceeds_quote and, when purchase_ts is supplied, estimated_arrival. The quote flag compares two point estimates; it is not a calibrated lateness probability. This Python interface exposes no HTTP API.

## Held-out examples and model packaging

Run the full pipeline followed by `PYTHONPATH=src python -m regression.verify` to generate `data/regression/deployment_orders.pkl` and `deployment_expected.csv` (locally generated, git-ignored). They contain 20 held-out test records with only input columns and IDs, and the expected predictions. `PYTHONPATH=src python -m regression.delivery` builds and tests a portable ZIP under `models/regression/`. Model weights, raw data and per-order records are not committed. Obtain the tested ZIP from the verified local run or reproduce it from the course data; do not deploy a new unverified fit under the old model label.

Pin versions from `environment.json` / `requirements-regression.lock.txt`. Package the `regression` module and its dependency modules under src so joblib can load the classes. Test all 20 predictions against the supplied expected values before publishing the app.

## Behaviour and limits

- Empty input, missing mandatory columns, negative prices/distances and malformed purchase dates raise clear ValueErrors.
- Unknown categories use the fitted encoder's fallback; missing optional numeric attributes use training-fitted imputation.
- Labels, actual receipt, final status and reviews are not estimator inputs. Never show an outcome to the prediction path.
- Show historical-held-out status, point-estimate limitations and long-tail uncertainty. No validated prediction interval is claimed.
- The deployed model is frozen as of June 1, 2018. No live order feed is needed. A prospective production model would need a separately governed retraining and monitoring process.
''')
    (docs/'中文版导读.md').write_text(f'''# Phase 3 配送回归导读

## 当前结论

本轮主模型为 `{winner}`，历史时间测试 MAE 为 {primary['mae']:.4f} 天，RMSE 为 {primary['rmse']:.4f} 天。以实际输出为准，不沿用 Phase 2 的随机测试数字。

## Notebook 阅读顺序

`notebooks/regression/delivery_regression_chronological.ipynb` 依次解释目标、原始数据核对、时间/标签边界、统一模型比较、时间与风险特征消融、ensemble 取舍、测试表现、长尾与路线误差、解释性分析、随机参考与部署交接。

`stacking_chronological.ipynb` 展示内部时间 OOF 和权重、外层比较及保留/舍弃依据。不会用旧随机 OOF 填充新比较。

## 必须保留的限制

gap 为 {g['q99_days']} 天，但同时强制每轮训练的标签已知；五月用于选择的可用标签仅覆盖约 65.3%，完整后验结果单独展示且不参与选型。课程数据已在历史开发中暴露，时间测试不是从未见过的外部数据。模型覆盖已观察完成的配送，不能把它解释为对未完成/取消订单的无条件预测。

报告和展示素材见本目录 `report_evidence_guide.md`、`presentation_notes.md`；部署交接见 `deployment_handoff.md`。成绩以 `outputs/regression/tables/` 和 `test_summary.json` 为准。
''')
    environment={'python':sys.version,'platform':platform.platform(),'versions':{}}
    import importlib.metadata
    for package in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','nbformat','nbclient','ipykernel']:
        environment['versions'][package]=importlib.metadata.version(package)
    data.save_json(C.OUT/'environment.json',environment)
    (C.ROOT/'requirements-regression.lock.txt').write_text('# Regression run: Python 3.12; separate from the classification environment.\n'+
        '\n'.join(f'{p}=={v}' for p,v in environment['versions'].items())+'\n')
    create_notebooks(notebooks,summary)
    (docs/'README.md').write_text('# Regression — Phase 3\n\n- [中文导读](中文版导读.md)\n- [Report evidence](report_evidence_guide.md)\n- [Decision log](decision_log.md)\n- [Presentation segment](presentation_notes.md)\n- [Deployment interface](deployment_handoff.md)\n- [Complete notebook](../../notebooks/regression/delivery_regression_chronological.ipynb)\n- [Stacking comparison](../../notebooks/regression/stacking_chronological.ipynb)\n')
    log('Report/presentation/deployment materials and notebook sources prepared.')

def catalogue():
    feature=table('feature_catalogue')
    feature=feature[~feature.feature.isin(['risk_p30','risk_unavailable'])].copy()
    definitions={
      'promised_lead_time_days':('orders','(estimated receipt - purchase timestamp) / 86400 seconds'),
      'max_distance_km':('items, orders, customers, sellers, geolocation','Maximum per-item haversine distance, rounded to 0.1 km; screened, deduplicated zip-prefix median coordinates'),
      'any_distance_missing':('same route join','1 if at least one item lacks a calculable distance'),
      'same_state':('customers, sellers, items','1 if every seller state equals the customer state'),
      'total_price':('order items','Sum item price per order'),
      'total_freight':('order items','Sum freight_value per order'),
      'freight_ratio':('order items','total_freight / total_price; nonpositive denominator becomes missing'),
      'item_count':('order items','Number of item rows'),
      'seller_count':('order items','Number of distinct sellers'),
      'primary_seller_price_share':('order items','Price of primary seller / total item price'),
      'seller_state_count':('order items, sellers','Number of distinct observed seller states'),
      'avg_product_weight_g':('order items, products','Mean positive item weight only when every item has valid weight; otherwise missing'),
      'any_weight_missing':('order items, products','1 if any item weight is missing or nonpositive'),
      'distinct_product_count':('order items','Number of distinct products'),
      'category_count_observed':('order items, products','Number of distinct nonmissing source category names'),
      'dominant_category_price_share':('order items, products','Highest summed category item price / total item price'),
      'max_product_weight_g':('order items, products','Maximum valid positive item weight'),
      'total_known_weight_g':('order items, products','Sum valid positive item weights; missing if none are known'),
      'max_product_volume_cm3':('order items, products','Maximum length * height * width, all three positive and observed'),
      'total_known_volume_cm3':('order items, products','Sum valid item volumes; missing if none are known'),
      'mean_unit_price':('order items','Mean item price'),
      'any_volume_missing':('order items, products','1 if at least one item lacks a valid positive three-dimensional volume'),
      'purchase_elapsed_days':('orders','Days since fixed 2016-01-01; diagnostic variant only'),
      'purchase_year_month':('orders','YYYY-MM categorical purchase date; diagnostic variant only'),
      'customer_state':('customers','customer_state joined by customer_id'),
      'customer_city':('customers','customer_city joined by customer_id'),
      'route_group':('items, customers, sellers','Unknown if states missing; otherwise any seller cross-state or all sellers same-state'),
      'primary_seller_id':('order items','Seller with greatest summed item price; seller_id lexical tie-break'),
      'primary_seller_state':('order items, sellers','State of the primary seller'),
      'primary_seller_city':('order items, sellers','City of the primary seller'),
      'primary_state_route':('customers, primary seller','primary seller state | customer state'),
      'primary_city_route':('customers, primary seller','seller state:city | customer state:city'),
      'primary_zip_zone_route':('customers, primary seller','Integer zip prefix // 1000 for seller and customer, concatenated'),
      'category_group':('products, category translation, items','Missing/incomplete, Multiple categories, or the first single English category'),
      'dominant_category':('products, items','Source category with highest summed price; lexical tie-break; missing category as explicit level')}
    for prefix,period,expr in [('month',12,'purchase month - 1'),('weekday',7,'purchase day of week (Mon=0)'),('hour',24,'purchase hour')]:
        for trig in ['sin','cos']:definitions[prefix+'_'+trig]=('orders',f'{trig}(2*pi*({expr})/{period})')
    for col in feature.feature:
        assert col in definitions,col
    feature['source']=feature.feature.map(lambda k:definitions[k][0])
    feature['formula_or_definition']=feature.feature.map(lambda k:definitions[k][1])
    for col,formula in [('risk_p30','P(duration >30 days) from an auxiliary classifier trained only on earlier monthly blocks'),
                        ('risk_unavailable','1 for early cold-start training rows without an earlier classifier; 0 otherwise')]:
        feature=pd.concat([feature,pd.DataFrame([{'feature':col,'group':'risk','type':'numeric',
            'availability':'Generated from purchase inputs by a previously fitted classifier',
            'source':'Auxiliary past-only risk model','formula_or_definition':formula,'target_derived':True,'used_in_main_pack':False}])],ignore_index=True)
    from .score import load_bundle
    selected_columns=set(load_bundle()['input_columns'])
    feature['used_in_selected_model']=feature.feature.isin(selected_columns)
    save_table(feature,'feature_catalogue')

def create_notebooks(folder,summary):
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    setup='''%matplotlib inline
from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display
root = next(q for p in [Path.cwd(), *Path.cwd().parents] for q in [p, p / "Phase_3"] if (q / "src/regression").is_dir())
sys.path.insert(0, str(root / "src"))
from regression import config as C
from regression.reporting import table, draw
from regression.pipeline import read
import matplotlib.pyplot as plt
pd.options.display.float_format = "{:,.4f}".format
print("Phase 3 regression — saved, frozen evidence")'''
    cells=[md('# Delivery Lead-Time Regression\n\n**IT5006 · Phase 3**\n\nAt order placement, estimate the fractional days until customer receipt. This notebook follows the project’s analytical sequence from raw tables and information availability through chronological validation, model comparison, interpretation and deployment. It reads frozen results; the complete training command is documented below.'),
      md('## Summary\n\n'+f"The chronological procedure retained **{summary['selected_candidate']}**. Test MAE was **{summary['test']['mae']:.4f} days** on **{summary['test_n']:,}** June–August purchases. Report both model error and the limitations of completed-delivery coverage, selection-label maturity and prior historical test exposure. These results do not establish prospective operational accuracy."),
      md('## Problem Scoping Checklist\n\n| Check | Assessment |\n|---|---|\n| Derivable target | Receipt minus purchase in fractional days from the fixed course tables. |\n| Realistic inputs | Purchase-time order/quote/product/location inputs; archived availability remains an assumption. |\n| Sufficient signal | Earlier EDA motivates predictors; the time evaluation tests their transfer to later orders. |\n| Skew and tail | Keep valid slow orders, use MAE and report tail magnitude and contribution. |\n| Stakeholder | Fulfilment and customer-service teams can use provisional estimates with human escalation. |'),
      md('## Context & Methods\n\nThe primary comparison uses the same feature schema and preprocessing for OLS, Ridge, one tree, RF and HGB. The target remains original days. Five expanding windows validate January–May 2018; the test contains June–August purchases. Training requires both an embargo and receipt before the fitting cutoff. Selection outcomes must be observed before 1 June. All choices freeze before test assessment.\n\n### Key assumptions\n\nThe source snapshot records completed outcomes and static metadata, not a full historical availability log. Observed completed deliveries form a conditional population. May has incomplete label maturity at selection time. The historical orders have been used in prior development. No live prospective evidence is claimed.'),
      code(setup),md('## 1. Data Sources and Order-Level Preparation\n\nSeven course tables are linked using documented identifiers, with line items aggregated before order modelling. The pre-model audit independently reconstructed targets, checked basket totals and verified that perturbing outcome fields leaves purchase features unchanged. The earlier V2 numeric fields were matched on every valid order.'),
      code('display(table("source_inventory"))\ndisplay(table("cohort_flow"))\ndisplay(pd.Series(read("audit_pass.json"), name="Pre-model audit"))'),
      md('## 2. Chronological Windows and Label Availability\n\nThe 39-day embargo is calibrated from outcomes already observed before the first deployment date. A strict receipt cutoff remains necessary: some longer labels cross even that gap. Validation membership is based on purchase month, not on completing within that month. Outcomes unavailable before final fitting are excluded from selection scores, with their coverage explicitly reported.'),
      code('display(table("fold_table"))\ndraw("01_windows"); plt.show()'),
      code('draw("09_label_coverage"); plt.show()\ndisplay(table("selection_maturity_sensitivity").head(10))'),
      md('**Interpretation.** May’s selection-label coverage is substantially lower. The January–April ranking and eventual-outcome diagnostic quantify this limitation; they do not trigger a new search after testing.'),
      md('## 3. Purchase-Time Features and Matched Model Comparison\n\nImputation, category pooling, encoding and scaling are fitted inside each training window. The primary model comparison changes the learner while holding inputs fixed. The candidate list and per-fold training results make the search inspectable.'),
      code('display(table("feature_catalogue"))\ndisplay(table("cv_family_summary"))\ndisplay(table("cv_all_candidates_folds").head(10))'),
      md('## 4. Calendar, Target Scale and Risk-Feature Checks\n\nFixed-HGB ablations separate cyclic calendar, absolute date and auxiliary risk inputs. The optional >30-day risk score uses earlier monthly training blocks, with labels observed before each block; early training rows carry an explicit unavailable-score flag. The RF log-target experiment holds its other conditions fixed.'),
      code('display(table("diagnostic_summary"))\ndraw("06_calendar_ablation"); plt.show()'),
      md('## 5. Ensemble Comparison\n\nOne convex Ridge/RF/HGB stack is compared. Every outer combiner is fitted using inner chronological predictions from its training partition. Base hyperparameters were previously selected using the main CV, so this is not a fully nested hyperparameter-selection estimate. The minimum retained gain is 1% versus the best single model.'),
      code('display(pd.Series(read("stacking_complete.json")))\ndraw("08_stack_weights"); plt.show()'),
      md('## 6. Frozen Train–Validation–Test Results\n\nThe test comparison list and selection hash were fixed before assessment. Training scores are in-sample; CV is an unweighted monthly mean with sample SD. The test is one later window. A discarded stack has no fabricated final Train/Test score.'),
      code('display(table("train_cv_test_comparison"))\ndisplay(pd.Series(read("test_summary.json")["success"]))\ndraw("02_model_comparison"); plt.show()'),
      md('## 7. Monthly, Route and Long-Tail Error\n\nAn average error does not imply every order is within that range. Duration groups form a disjoint partition, enabling an error-contribution calculation. Route and chronology slices overlap and must not be added together.'),
      code('display(table("test_by_month"))\ndraw("03_test_months"); plt.show()'),
      code('display(table("test_error_groups"))\ndraw("04_tail_errors"); plt.show()\ndisplay(table("test_by_route").head(10))'),
      md('## 8. Validation-Based Feature Reliance\n\nGroups of related input columns are shuffled jointly on May outcomes observed before final selection. Risk is recomputed from perturbed source inputs when applicable. The change in MAE measures predictive reliance, not a causal effect. Repeat SD describes permutation variability.'),
      code('display(table("permutation_importance_groups"))\ndraw("05_importance"); plt.show()'),
      md('## 9. Random Reference and Outcome-Maturity Diagnostic\n\nThe random reference uses the chronological single-model specification without tuning it again. Training eligibility, data mix and test populations differ, so its score difference is not isolated algorithm improvement. Eventual validation outcomes are examined only after freezing, without feeding model selection.'),
      code('display(table("chronological_vs_random"))\ndraw("07_random_reference"); plt.show()\ndisplay(table("validation_outcome_maturity_diagnostic"))'),
      md('## 10. Deployment and Reproducibility\n\n`regression.score.predict_lead_time` returns predicted days, quoted days, a point-estimate comparison flag and an estimated arrival timestamp. The app must use the frozen bundle and held-out historical inputs. See `docs/regression/deployment_handoff.md` for exact inputs, errors and golden predictions.\n\nA fresh run, from `Phase_3`, uses:\n```bash\nOLIST_ARCHIVE=/path/to/IT5006_Project-Data.zip python src/regression/run_pipeline.py --stage all --output-root /path/to/new_run\n```\nThe reading notebook itself does not retrain models or alter the selected results.'),
      md('## Conclusions and Limitations\n\nThe revised evaluation addresses future-order use and makes label availability explicit. Retain the actual success/failure against simple references, the monthly variation and slow-order limitations. May’s incomplete selection labels, the conditional completed-delivery cohort, a short temporal record and prior exposure prevent treating this as untouched prospective confirmation. Deployment supports a demonstration; business benefit requires a separate pilot.\n\n**AI declaration:** Codex assisted with code, checks, visualisation and writing. Numerical results are computed from the saved run; the team remains responsible for the interpretation and submission.')]
    # Keep the submission reading path compact; full audit/search data remain in the linked CSVs.
    for cell in cells:
        if cell.cell_type=='code':
            s=cell.source
            if 'table("source_inventory")' in s:
                cell.source='''display(table("source_inventory")[["table", "file", "rows"]])
display(table("cohort_flow"))
audit = read("audit_pass.json")
checks = ["zero_future_training_labels", "zero_unavailable_selection_labels", "train_score_disjoint", "test_excluded_from_development"]
display(pd.DataFrame({"Check": checks, "Passed": [audit[k] for k in checks]}))'''
            elif 'table("fold_table")' in s:
                cell.source='''display(table("fold_table")[["fold", "cutoff", "train_n", "score_n", "score_label_coverage", "labels_at_or_after_cutoff"]])
draw("01_windows"); plt.show()'''
            elif 'table("feature_catalogue")' in s:
                cell.source='''catalogue = table("feature_catalogue")
display(catalogue.groupby("group").agg(features=("feature", "size"), selected_features=("used_in_selected_model", "sum")))
display(table("cv_family_summary")[["family", "candidate", "train_mae", "cv_mae", "cv_sd"]])
display(table("cv_all_candidates_folds")[["candidate", "fold", "train_mae", "val_mae"]].head(10))'''
            elif 'table("diagnostic_summary")' in s:
                cell.source='''display(table("diagnostic_summary")[["candidate", "train_mae", "cv_mae", "cv_sd", "cv_rmse"]])
draw("06_calendar_ablation"); plt.show()'''
            elif 'pd.Series(read("stacking_complete.json"))' in s:
                cell.source='''stack = read("stacking_complete.json")
display(pd.DataFrame({"Measure": ["Stack CV MAE", "Best single CV MAE", "Relative gain", "Retained"],
                      "Value": [stack["cv_mae"], stack["best_single_cv_mae"], stack["relative_gain"], stack["retained"]]}))
draw("08_stack_weights"); plt.show()'''
            elif 'table("train_cv_test_comparison")' in s:
                cell.source='''display(table("train_cv_test_comparison")[["candidate", "train_mae", "cv_mae", "cv_sd", "test_mae"]].fillna("—"))
display(pd.Series(read("test_summary.json")["success"]))
draw("02_model_comparison"); plt.show()'''
            elif 'table("test_error_groups")' in s:
                cell.source='''display(table("test_error_groups")[["group", "n", "mae", "bias", "order_share", "absolute_error_share"]])
draw("04_tail_errors"); plt.show()
display(table("test_by_route").head(10))'''
            elif 'table("chronological_vs_random")' in s:
                cell.source='''comparison = table("chronological_vs_random")[["n", "mae", "rmse", "r2"]].copy()
comparison.insert(0, "Procedure", ["Chronological main", "Random reference"])
display(comparison)
draw("07_random_reference"); plt.show()
display(table("validation_outcome_maturity_diagnostic")[["fold", "scope", "n", "mae", "rmse"]])'''
        elif cell.source.startswith('## Summary'):
            cell.source='## Summary\n\n'+f"The chronological procedure retained **{summary['selected_candidate']}**. Test MAE was **{summary['test']['mae']:.4f} days** on **{summary['test_n']:,}** June–August purchases. Error varies by month, route and delivery duration. Completed-delivery coverage, selection-label maturity and prior historical test exposure limit the interpretation; these results do not establish prospective operational accuracy."
        elif cell.source.startswith('## 3. Purchase-Time'):
            cell.source+='\n\nThe [complete feature catalogue](../../outputs/regression/tables/feature_catalogue.csv) gives source tables, formulas, availability assumptions and selected-model usage for every input. The [full search evidence](../../outputs/regression/tables/cv_all_candidates_folds.csv) retains all candidate and fold scores.'
        elif cell.source.startswith('## Conclusions and Limitations'):
            cell.source='## Conclusions and Limitations\n\n'+f"The retained model meets both pre-set historical criteria: test MAE is {summary['test']['mae']:.4f} days, {summary['success']['mae_reduction_vs_median']:.1%} below the training-median reference, and lower than the quoted-promise error. Individual slow-order errors remain large. The risk-HGB advantage over Ridge in selection is small and sensitive to May coverage, so it does not establish stable superiority.\n\nMay’s incomplete selection labels, the conditional completed-delivery cohort, a short temporal record and prior exposure prevent treating this as untouched prospective confirmation. Deployment supports a demonstration; business benefit requires a separate pilot.\n\n**AI declaration:** Codex assisted with code, checks, visualisation and writing. Numerical results are computed from the saved run; the team remains responsible for interpretation and submission."
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    write_notebook(nb,folder/'delivery_regression_chronological.ipynb')
    st=[md('# Chronological Ensemble Comparison\n\n**IT5006 · Phase 3 — delivery lead time**\n\nThis companion records the single convex ensemble experiment. It shares the regression cohort and outer windows, rather than reusing random Phase 2 OOF predictions.'),code(setup),
        md('## Nested Temporal Construction\n\nWithin each outer training set, the last three available purchase months provide inner out-of-time predictions. Each base fit uses earlier purchases, the embargo and labels known at that inner cutoff. Meta labels must be known before the outer cutoff. Nonnegative weights sum to one and the intercept is zero. Base configurations were selected by the main outer CV, which limits the independence of this comparison.'),
        code('display(table("stacking_inner_audit"))'),md('## Outer Validation and Weights'),
        code('display(table("stacking_folds"))\ndraw("08_stack_weights"); plt.show()'),
        md('## Retention Decision\n\nA 1% reduction in mean validation MAE is required over the best single model. Discarded ensembles do not receive a final test score. No test result is used to revise weights or select another combination.'),
        code('display(pd.Series(read("stacking_complete.json")))\ndisplay(table("train_cv_test_comparison"))'),
        md('## Reproducibility\n\nUse the main regression pipeline through its `stacking` and `freeze` stages; raw data, inner predictions and fitted estimators are generated locally. The saved reading outputs are sufficient to inspect the reported comparison. See `docs/regression/report_evidence_guide.md` for interpretation and limitations.')]
    write_notebook(nbformat.v4.new_notebook(cells=st,metadata=nb.metadata),folder/'stacking_chronological.ipynb')

def write_notebook(nb,path):
    if path.exists():
        import copy
        old=nbformat.read(path,as_version=4)
        for before,after in zip(old.cells,nb.cells):
            if before.cell_type==after.cell_type and before.source==after.source:
                after.id=before.id;after.metadata=copy.deepcopy(before.metadata)
                if before.cell_type=='code':
                    after.outputs=copy.deepcopy(before.outputs);after.execution_count=before.execution_count
        nb.metadata=copy.deepcopy(old.metadata)
    nbformat.write(nb,path)

def run(stage):
    if stage=='figures':return figures()
    if stage=='package':return package()
    raise ValueError('Unknown stage '+stage)
