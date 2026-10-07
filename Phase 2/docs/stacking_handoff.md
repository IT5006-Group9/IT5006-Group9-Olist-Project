# Delivery Regression · Stacking Input Handoff

**Ensemble entry-point update: 2026-10-07.** The latest [Notebook](../notebooks/ensemble_comparison.ipynb) and [reproduction guide](ensemble_comparison.md) present the Ridge + forest 50/50 average and constrained MAE stacking together. New artifacts are located in `versions/ensemble_comparison/`. The base models use the audited v2 inputs and previously retained parameters; older handoff files cannot substitute for these artifacts. Author: **yexueying70-cell**.

The original handoff record is preserved below. Statements such as “not fitted” or “not tested” refer to the status at that time.

---

**最新特征/回测更新（2026-10-05）：** [特征优化说明](feature_optimization.md)和[已运行Notebook](../notebooks/feature_optimization.ipynb)记录新增18组配置，未找到进一步收益；保留上一轮Ridge/森林，三月MAE6.847/6.913天。恢复3,669笔慢订单后成熟历史CV5.236/5.219天；旧约4.54天须标为条件化口径。以下原交接/报告/导读保留历史语境，不能与新版本混用，最终测试未评分。

**交接状态更新（2026-10-05）：** [独立模型复核](baseline_variant_review.md)已在审计v2上比较原方案和有限改进，结果保存在`versions/baseline_review_v2/`。以下正式交接仍是v1历史文件；新方案冻结之前不能将两版数据、pipeline或OOF混用。复核未拟合stacking或评分最终测试。

**修正已落实：** 正式`prepare_data`默认已改为筛查异常坐标、去重复参考点、邮编中位数；独立生成并核对的v2输入位于`versions/preparation_v2/data/eligible_orders.csv`。原始ZIP、旧`data/eligible_orders.csv`和模型/OOF/预测保留。下一轮模型使用v2并统一更新全部结果；当前6–7天MAE仍属于v1，没有重训或重新评分。下文候选探索记录保留检查过程，以此修正状态为准。


**2026-10-05 数据复核更新：以下模型结果与交接文件使用旧邮编均值输入。有效订单/目标保持，地理处理已修正并生成v2；`notebooks/data_preparation_audit.ipynb`已运行核对。下一轮使用v2重训及重生成OOF之后才能定稿交接。当前审计没有模型拟合或测试评分。**


本交接为本地开发阶段材料，**没有拟合 stacking，没有最终测试分数**。原始数据和逐订单预测未上传。

## 开发数据与模型

| 内容 | 本地位置 | 用途 |
|---|---|---|
| 确切字段、公式和输入角色 | `config/feature_roles.csv` | 16 个实际输入；不能加入结果字段 |
| 完整合格订单与时间戳 | `data/eligible_orders.csv` | 数据来源与后续统一复现 |
| 全部划分及逐折拟合/预测 ID | `data/split_manifest.csv`、`data/cv_fold_manifest.csv` | 固定同样本评价、审计标签可见性 |
| 开发期 X/y | `data/stacking_train.csv`、`data/stacking_validation.csv` | 训练与 3 月验证；两者分开 |
| 对齐 OOF | `outputs/oof/base_oof_predictions.csv` | 元模型训练输入，39,445 行；含订单 ID、目标、fold 和各基模型预测 |
| 独立验证预测 | `outputs/predictions/validation_predictions.csv` | 在 7,003 笔 3 月订单上比较组合与单模型 |
| 拟合 pipeline | `models/*_development.joblib` | 使用 53,644 笔截止 2018-03-01 已完成订单拟合 |
| 配置/依赖/结果 | `outputs/selected_models.json`、`requirements.txt`、`outputs/tables/` | 模型参数、种子5006、数据指纹和误差 |
| 锁定测试输入 | `data/locked_test_features.csv` | 12,507 笔 7–8 月订单，仅输入和订单 ID；当前不做最终评分 |

OOF 列为 `linear_oof_days`、`ridge_oof_days`、`tree_oof_days`、`forest_oof_days`，均为天数；`lead_time_days` 是元模型训练目标。前14,199笔预热训练订单无有效历史 OOF，不进入元模型训练。所有 OOF 行具有共同折编号和所有基模型预测；不要按行位置盲目连接，要用 `order_id`。

Plain LinearRegression 和 Ridge 属于同一个线性家族；单树和随机森林属于树家族。加入 stacking 时对应可选第三家族，不能把这些算法各算成一个家族。组合器可在开发期选择必要输入，勿因有四列就默认全部必须加入。

## 正确的使用顺序

1. 用开发 OOF 预测列及其目标拟合元模型；需要选择元模型参数时，使用 OOF 中的时间折，不能用拟合预测替代。
2. 基模型已在完整开发训练集拟合；其 3 月验证预测供独立验证组合效果。元模型训练不能包含这批验证订单的目标。
3. 与相同 7,003 笔订单的单模型/参照表比较 MAE、RMSE、R²、偏差和尾部误差；不能直接拿 CV 均值与验证集分数比较提升百分比。
4. 基模型、组合器及任何变换方案固定后，再统一用截至 2018-07-01 已知目标的历史订单重新拟合。若扩展元模型训练到完整预测试期，也必须重建相应时间 OOF，不能把旧训练集的拟合预测补进去。
5. 所有最终模型评价同一锁定测试集，保留少量 >60 天样本的数量与限制；不能依据这些分数反复调整组合器。

## 加载与推断

从项目根目录执行，使用本项目环境与同一 schema：

```python
import sys
import joblib
import pandas as pd
sys.path.insert(0, "src")
from delivery_regression import FEATURES, predict_days

validation = pd.read_csv("data/stacking_validation.csv")
model = joblib.load("models/forest_development.joblib")
prediction_days = predict_days(model, validation[FEATURES])
```

`predict_days` 将负的数学点预测限制为0，定义为预测时已知的物理边界，全部实验采用相同规则。日历字段是整数域的类别，地理/品类字段是字符串；不要把月份变成“2018-03”或把商品数当品类。模型预处理已包含填补、缩放和编码，不能外部再全量拟合这些步骤。

## 重要边界

- 当前 OOF 来自有限候选配置（开发阶段补充Ridge），列选择只使用训练 CV。它用于元模型拟合，不是无偏最终性能估计；配置选择与最终评价的分工须保留。
- 训练折要求实际收货早于预测起点。按购买日期隔开，却把尚未出现的标签拿来训练，仍会泄漏。
- 截止 3 月 1 日尚未完成的早期订单有3,673笔，它们没有被判成脏数据或永久删掉，只是不进入当时的训练；后续完整预测试期重新拟合时需重新判断可见性。
- 4–6月19,643笔作为标签成熟间隔，不参与当前3月模型选型；后续统一预测试期重新拟合可纳入当时已完成订单。
- 七月截止确保本次3月验证的37笔 >60天订单也已可观察。最终测试仅有6笔 >60天订单，不能据此做强尾部泛化结论。
- 原始归档承诺是否被修改、邮编中心代理、完成配送选择偏差和未知物流冲击仍是限制。输出是时长估计，不是经校准的服务承诺区间。

本项目 Notebook、Python 源码和运行说明构成交接的可复现部分；数据、OOF 与拟合文件需另按明确共享授权提供。
