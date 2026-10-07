"""Build the concise, results-first publication notebook for two ensembles."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(r"""# 配送时长预测：简单平均与约束 MAE Stacking

**作者：yexueying70-cell**

本实验在既有四类单模型的基础上，保留两个集成方案：**Ridge＋随机森林的50/50简单平均**，以及**通过历史 OOF 学习权重的约束 MAE stacking**。简单平均作为不学习组合权重的参照，严格来说不是 stacking。

主要问题是：学习组合权重，能否带来值得增加模型复杂度的收益？历史七个月回测中，约束 stacking 的平均 MAE 为 **5.724天**，50/50平均为 **5.717天**，二者非常接近，尚未显示学习权重具有明确优势。其他组合尝试仅在文末用汇总表保留。

**评价边界：原最终测试成绩已在此前查看，本轮属于受到既有测试信息影响的探索性开发。** 本轮仅复现两个固定方案的历史月度回测，不生成或评价七月起的测试预测，不能据此宣称新的独立测试提升。

本 Notebook 默认只读仓库中的公开汇总，可直接运行；逐订单数据、OOF、预测和训练模型不随仓库上传。
"""),
        code('''from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next((path for path in [Path.cwd(), *Path.cwd().parents]
             if (path / "src/delivery_regression.py").is_file()), None)
if ROOT is None and (Path.cwd() / "Phase 2/src/delivery_regression.py").is_file():
    ROOT = Path.cwd() / "Phase 2"
if ROOT is None:
    raise FileNotFoundError("请从项目根目录、Phase 2 或其 notebooks 目录运行。")
sys.path.insert(0, str(ROOT / "src"))
DEST = ROOT / "versions/ensemble_comparison"

# 默认只读已发布的摘要。完整重跑需要课程原始 CSV，以及明确替换本轮输出。
RUN_EXPERIMENT = False
CSV_DIR = None  # 例如 Path("/path/to/Olist_CSV")；请勿提交个人绝对路径。
if RUN_EXPERIMENT:
    if CSV_DIR is None:
        raise ValueError("先设置 CSV_DIR，再开启 RUN_EXPERIMENT。")
    subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_ensemble_comparison.py"),
                    "--csv-dir", str(CSV_DIR), "--overwrite"], cwd=ROOT, check=True)

def table(name):
    return pd.read_csv(DEST / "outputs/tables" / name)

def read_json(name):
    return json.loads((DEST / "outputs" / name).read_text())

required = ["main_summary.csv", "monthly_metrics.csv", "coverage_by_month.csv",
            "final_weights.csv", "initial_model_comparison.csv", "other_attempts.csv",
            "prior_test_comparison.csv"]
missing = [name for name in required if not (DEST / "outputs/tables" / name).exists()]
if missing:
    raise FileNotFoundError("缺少已发布摘要：" + ", ".join(missing))
protocol = read_json("experiment_protocol.json")
result = read_json("run_result.json")
assert result["test_scored"] is False
assert result["test_predictions_generated"] is False
summary, monthly = table("main_summary.csv"), table("monthly_metrics.csv")
METHODS = ["linear", "ridge", "tree", "forest", "mean_ridge_forest", "convex_two_full_history"]
LABELS = {"linear": "普通线性回归", "ridge": "改进 Ridge", "tree": "决策树",
          "forest": "近期加权随机森林", "mean_ridge_forest": "Ridge＋森林 50/50平均",
          "convex_two_full_history": "Ridge＋森林约束 MAE stacking"}
print("公开结果目录：", DEST.relative_to(ROOT))
print("本轮生成测试预测：", result["test_predictions_generated"])
print("本轮评分测试集：", result["test_scored"])
'''),
        md(r"""## 1. 从四类单模型出发

既有模型包括普通线性回归、Ridge、决策树及随机森林。本轮复用原研究中已确定的输入特征与配置，不重新搜索基础模型参数。

| 模型 | 本轮月度比较使用的版本 | 与初始配置的关系 |
|---|---|---|
| 普通线性回归 | `linear` | 保留原版本 |
| Ridge | `ridge_logx_a1000` | 对部分数值输入作 log 变换，α=1000；初始 Ridge 为 α=100 |
| 决策树 | `tree_d8` | 保留深度8版本 |
| 随机森林 | `rf_recent180` | 保留近期加权版本，权重半衰期180天；不是只取最近180天的数据 |

因此，后文简写的 Ridge、森林都指**改进后的版本**，不能把对应成绩标为原始参数成绩。下表单独保存原有单模型的三月验证记录，作为实验起点；它与后面的七个月平均 MAE 不是同一种汇总。
"""),
        code('''print("历史单模型比较：以表内评价范围为准，不与七个月均值直接比较。")
initial = table("initial_model_comparison.csv")
display(initial[["model_label", "configuration", "n", "MAE_days", "RMSE_days", "period"]]
        .rename(columns={"model_label": "模型", "configuration": "配置", "n": "订单数",
                         "MAE_days": "MAE天", "RMSE_days": "RMSE天", "period": "评价范围"}).round(4))
'''),
        md(r"""## 2. 时间验证与可见标签

目标是下单至收货的连续天数。每个预测月份的起点为 $t$，基础模型只使用 **下单时间 < $t$ 且收货时间 < $t$** 的订单训练，预处理也在该训练集内拟合。历史每月生成当月 OOF（out-of-fold）预测；训练元模型时还要求其 OOF 预测起点早于当前 $t$，避免使用同月或未来标签。

| 时间范围 | 用途 |
|---|---|
| 2017年7—9月 | 累积 OOF，作为元模型预热 |
| 2017年10月—2018年4月 | 七个主验证月份；报告每月 MAE 的等权平均 |
| 2018年5月 | 固定方案后的敏感性检查；不参与原方案选择 |
| 2018年6月 | 不评分；其中已可见的 OOF 标签可以用于最终七月前历史拟合 |

所有历史评分目标还必须在 **2018-07-01 前收货**。六月仍有约27.44%的合格订单标签不可见，不能当作完整验证月份。覆盖率的分母是数据中最终可识别的合格订单，不代表平台全部订单；未可见标签不计为零误差。这里假定每月重训，不能直接代表固定两个月不重训的部署方式。
"""),
        code('''coverage = table("coverage_by_month.csv")
display(coverage[["month", "role", "eligible_n", "known_by_july_n", "pending_n", "pending_pct"]]
        .rename(columns={"role": "用途", "eligible_n": "合格订单数", "known_by_july_n": "七月前标签可见数",
                         "pending_n": "仍不可见数", "pending_pct": "仍不可见百分比"}).round(3))
'''),
        md(r"""## 3. 只保留两个集成方案

**方案 A：50/50简单平均。** 两个基础模型给出以天为单位的非负预测，直接取平均，无需训练元模型：

$$\hat y_{\mathrm{mean}}=0.5\hat y_{\mathrm{Ridge}}+0.5\hat y_{\mathrm{Forest}}.$$

**方案 B：约束 MAE stacking。** 在起点前全部符合标签可见性条件的历史 OOF 上学习权重：

$$\min_{w_R,w_F}\frac{1}{n}\sum_{i=1}^{n}\left|y_i-w_R\hat y_{i,R}-w_F\hat y_{i,F}\right|,
\qquad w_R,w_F\geq0,\quad w_R+w_F=1,\quad b=0.$$

实现通过线性规划直接优化绝对误差。权重作用于原始“预测天数”，截距固定为0；组合值位于两个基础预测之间。它能调整组合比例，但若两个基础模型同时犯同方向错误，这种约束不能保证纠正该错误。

下方函数展示实际使用的实现；完整复现脚本调用同一个 `ConvexMAERegressor`。默认展示已保存结果，不在打开 Notebook 时训练。最终权重来自2018年7月1日前的历史拟合；各验证月独立拟合各自权重，不能用最终权重回填过去月份。
"""),
        code('''from stacking_tuning import ConvexMAERegressor, meta_fit_rows

META_COLUMNS = ["ridge_oof_days", "forest_oof_days"]

def predict_equal_average(base_predictions):
    """基础预测已按 order_id 对齐，且单位为天。"""
    return base_predictions[META_COLUMNS].mean(axis=1).to_numpy()

def fit_constrained_stacking(historical_oof, origin):
    """复用严格时间与标签筛选；全部符合条件的历史，不设置额外窗口。"""
    fit = meta_fit_rows(historical_oof, pd.Timestamp(origin), window_days=None, gap_days=0)
    model = ConvexMAERegressor().fit(fit[META_COLUMNS], fit["lead_time_days"])
    assert np.all(model.coef_ >= 0) and np.isclose(model.coef_.sum(), 1)
    assert model.intercept_ == 0
    return model

weights = table("final_weights.csv")
display(weights[["candidate", "input", "weight_in_days_space", "intercept_days", "weight_sum"]]
        .replace({"candidate": LABELS}).round(5))
print("这是最终历史拟合的权重；重训时重新学习，不将其写成永久固定配比。")
'''),
        md(r"""## 4. 在相同月份、相同订单上比较

主指标为 **七个月 MAE 的等权平均**，单位天，越低越好。最差月份 MAE、平均月度 RMSE、月度偏差绝对值的平均用于观察取舍。`bias = 预测 − 实际`，负值表示低估。

下表在同一评价范围内比较四个保留的基础模型及两个集成；五月单列，不混入主指标。样本更多的月份不会在主指标中获得更大权重。
"""),
        code('''main = summary.loc[summary["candidate"].isin(METHODS)].copy()
primary = main.loc[main.role.eq("primary")].set_index("candidate").loc[METHODS]
sensitivity = main.loc[main.role.eq("sensitivity")].set_index("candidate").loc[METHODS]
comparison = primary[["MAE_month_mean", "MAE_worst_month", "RMSE_month_mean", "bias_abs_month_mean"]].copy()
comparison["May_MAE_days"] = sensitivity["MAE_month_mean"]
comparison.index = comparison.index.map(LABELS)
comparison = comparison.rename(columns={"MAE_month_mean": "主验证平均MAE", "MAE_worst_month": "最差月份MAE",
    "RMSE_month_mean": "月度平均RMSE", "bias_abs_month_mean": "平均月度绝对偏差", "May_MAE_days": "五月MAE"})
display(comparison.round(4))

learned = primary.loc["convex_two_full_history", "MAE_month_mean"]
equal = primary.loc["mean_ridge_forest", "MAE_month_mean"]
forest = primary.loc["forest", "MAE_month_mean"]
print(f"约束 stacking 相对森林的 MAE 降幅：{100 * (forest - learned) / forest:.2f}%")
print(f"约束 stacking − 50/50平均：{learned - equal:+.6f} 天；正值表示简单平均更好。")
print("该差距仅是历史验证点估计，不是统计显著性或等效性检验。")
'''),
        code('''# 图中文字使用英文以兼容无中文字体的执行环境；完整中文解释见上表。
plot_names = {"linear": "Linear", "ridge": "Ridge (reviewed)", "tree": "Tree",
              "forest": "Forest (recency weighted)", "mean_ridge_forest": "50/50 average",
              "convex_two_full_history": "Constrained MAE stack"}
colors = {"ridge": "#9A6A36", "forest": "#6D7D99", "mean_ridge_forest": "#16817A",
          "convex_two_full_history": "#CB5B3D"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})
values = primary.loc[METHODS, "MAE_month_mean"]
axes[0].barh([plot_names[x] for x in METHODS], values,
             color=[colors.get(x, "#B7BDC5") for x in METHODS])
axes[0].invert_yaxis()
axes[0].set_xlim(0, float(values.max()) + .8)
for row, value in enumerate(values):
    axes[0].text(value + .05, row, f"{value:.3f}", va="center", fontsize=9)
axes[0].set_title("Primary validation: equal-month MAE")
axes[0].set_xlabel("MAE (days; lower is better)")
months = sorted(monthly["month"].unique())
for method in ["ridge", "forest", "mean_ridge_forest", "convex_two_full_history"]:
    curve = monthly.loc[monthly.candidate.eq(method)].set_index("month").loc[months]
    axes[1].plot(np.arange(len(months)), curve.MAE_days, marker="o", markersize=4,
                 label=plot_names[method], color=colors[method],
                 linestyle="--" if method == "mean_ridge_forest" else "-")
axes[1].axvspan(len(months)-1.5, len(months)-.5, color="#DFE3E7", alpha=.55)
axes[1].set_xticks(np.arange(len(months)), months, rotation=40, ha="right")
axes[1].set_title("Monthly MAE; shaded May = sensitivity")
axes[1].set_ylabel("MAE (days)")
axes[1].legend(fontsize=8, frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax is axes[0] else "y", alpha=.15)
fig.tight_layout()
plt.show()
plt.close(fig)
'''),
        md(r"""## 5. 结论与限制

约束 stacking 比近期加权森林的主验证 MAE 约低 **0.41%**，但没有优于50/50简单平均；最终学习比例也接近各一半。当前证据支持把简单平均作为低复杂度的集成参照，把约束 stacking 作为完整保留的学习权重实验，**不支持宣称 stacking 获得明确、稳定的额外收益**。

五月只用于敏感性检查。在该月，普通线性回归优于这两个组合，因此组合也不是每月都最好的方案。平均分改善并不代表所有月份、所有误差指标都改善。

结果仍受以下因素限制：基础配置来自先前非嵌套的筛选；既有调优重复使用过主验证月份；原测试结果已经影响后续开发方向；截至七月仍存在未成熟标签和已完成配送样本条件。需要新的、未参与开发的未来数据，才能评价新组合的独立泛化收益。
"""),
        md(r"""## 附录 A：其他尝试，仅保留结果摘要

此前还试过四模型 Ridge 元模型、四模型约束 MAE、中位数回归、元模型最近180天窗口和30天下单间隔。下表保留这些尝试的结果与选择理由；本发布入口不重新运行全套搜索。两阶段选择曾以七个主验证月的等权 MAE 为依据，对距离最低值不超过0.01天的方案按预设简单性顺序选择，最终保留两模型约束与全部历史。0.01天是操作性阈值，不是统计等效性界限。

表中的评价范围区分三月单月验证和七个月平均 MAE，不跨范围排名。月度 Ridge stacking 是在月度 OOF 上重新拟合的版本；它与首版 stacking 的 OOF 划分不同，不能把两者的分数接成同一轮提升。
"""),
        code('''attempts = table("other_attempts.csv")
display(attempts[["model_label", "evaluation", "n", "MAE_days", "decision"]]
        .rename(columns={"model_label": "尝试方案", "evaluation": "评价范围", "n": "订单数",
                         "MAE_days": "MAE天", "decision": "保留理由或处理"}).round(4))
'''),
        md(r"""## 附录 B：原最终测试的历史记录

首版四模型 Ridge stacking 在三月验证的 MAE 为6.558天，但在此前已评价的原最终测试上为4.916天；普通线性回归在该测试上为3.761天。三月的优势没有延续，这也是后续探索更简单组合的背景。

下表只是保留既有测试记录；**本 Notebook 没有重新评分该测试集，新约束 stacking 也没有新的独立测试成绩**。原测试与本轮月度重训的评价范围、运行方式不同，不能用两个表直接作提升率比较。
"""),
        code('''prior_test = table("prior_test_comparison.csv")
display(prior_test[["model_label", "n", "MAE_days", "RMSE_days", "period"]]
        .rename(columns={"model_label": "历史模型", "n": "订单数", "MAE_days": "MAE天",
                         "RMSE_days": "RMSE天", "period": "评价范围"}).round(4))
'''),
        md(r"""## 复现与核验

从 `Phase 2` 目录运行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-stacking.txt
.venv/bin/python scripts/reproduce_ensemble_comparison.py --csv-dir /path/to/Olist_CSV
.venv/bin/python scripts/verify_ensemble_comparison.py
.venv/bin/python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb
```

也可用 `--archive /path/to/IT5006_Project-Data.zip` 提供课程压缩包。已有本轮本地训练输出时，加 `--overwrite` 明确重建。完整重跑只训练这里的两个固定方案及基础参照，不搜索其他 stacking，也不生成七月起测试预测。

下面校验已发布协议中的源文件哈希及当前、历史摘要的哈希，并展示保存的审计状态。这属于**公开摘要与缓存记录检查**；逐订单对齐、时间界限、重新计算误差及模型回放，需要本地数据与模型存在，并运行上面的独立核验脚本。仅运行此 Notebook 不能替代完整审计。没有本地原始数据时，可运行 `scripts/verify_ensemble_comparison.py --summary-only` 进一步检查公开汇总的一致性。
"""),
        code('''def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_checks = []
for relative_path, expected in protocol.get("source_sha256", {}).items():
    path = ROOT / relative_path
    source_checks.append({"file": relative_path, "exists": path.is_file(),
                          "matches_saved_sha256": path.is_file() and sha256(path) == expected})
if not source_checks:
    raise ValueError("发布协议没有源文件哈希，无法验证结果对应的代码。")
display(pd.DataFrame(source_checks))
assert all(row["matches_saved_sha256"] for row in source_checks), "源文件与已发布协议不一致。"

public_hashes = result["public_artifact_sha256"]
for relative_path, expected in public_hashes.items():
    assert sha256(DEST / relative_path) == expected, f"公开摘要已改变：{relative_path}"
historical = read_json("historical_evidence.json")
for relative_path, expected in historical["published_exports_sha256"].items():
    assert sha256(ROOT / relative_path) == expected, f"历史摘要已改变：{relative_path}"
assert historical["new_test_scores_generated"] is False
print(f"当前实验公开摘要校验：{len(public_hashes)} 项；历史导出校验：{len(historical['published_exports_sha256'])} 项。")

audit_path = DEST / "outputs/validation_audit.json"
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    compact_audit = {key: value for key, value in audit.items()
                     if key in ["status", "success", "passed", "checks_passed", "test_scored",
                                "test_predictions_generated", "source", "mode", "full_local_audit"]}
    print("保存的审计摘要（本 cell 未重新执行完整审计）：")
    print(json.dumps(compact_audit or {"available": True}, ensure_ascii=False, indent=2))
else:
    print("未提供缓存审计记录；不能据此认定完整核验通过。")
print("Notebook 当前完成：公开摘要读取、模型约束展示、可视化与源文件/摘要哈希检查。")
'''),
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "authors": [{"name": "yexueying70-cell"}],
    }
    nbf.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "ensemble_comparison.ipynb", "exec")
    destination = ROOT / "notebooks/ensemble_comparison.ipynb"
    nbf.write(notebook, destination)
    print(f"Built {destination.relative_to(ROOT)} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    build()
