# Phase 2 · Delivery Lead-time Regression

## Ensemble comparison: one entry point, two methods

**Updated: 2026-10-07**

Read the [ensemble Notebook](notebooks/ensemble_comparison.ipynb) for the complete sequence: four original models → Ridge + random forest 50/50 average → two-model constrained MAE stacking → comparison on the same periods and conclusions. The [method and reproduction guide](docs/ensemble_comparison.md) provides execution commands. Other stacking attempts are collected in an appendix summary table instead of separate experiment Notebooks.

| Retained ensemble method | Mean monthly MAE, October 2017–April 2018 | May 2018 MAE, sensitivity check only |
|---|---:|---:|
| Ridge + forest 50/50 average | **5.717 days** | **4.574 days** |
| Ridge + forest constrained MAE stacking | 5.724 days | 4.593 days |

The base models are the previously retained Ridge with log-transformed inputs and the forest with recency weighting. Constrained stacking learns nonnegative weights summing to one from temporal OOF predictions, with a fixed zero intercept. The final historical weights are approximately 53.1%/46.9%. Learning weights has not demonstrated a clear benefit over simple averaging.

This is exploratory development performed after the first stacking version's test results had been inspected. The Notebook retains that version's weaker test performance and separates results by evaluation period. The new methods have no new independent test scores. Every historical fit uses only orders purchased and delivered before its origin, and preprocessing is fitted within each training fold.

The public files include the executed Notebook, source code, summary tables, verification records, and pinned dependencies. Raw data, order-level data, OOF predictions, scoring predictions, and model files remain local. The new entry point can reproduce the comparison independently from the course CSVs or ZIP, without local older stacking Notebooks or older experiment directories.

The earlier single-model and feature experiments are preserved below. Statements such as “stacking not fitted” and “test not scored” refer only to their respective historical stages.

---

**小组共享入口（2026-10-07）：** 本目录为[小组仓库](https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project)根目录的`Phase 2/`配送回归开发审阅成果，复制自独立工程提交`0625de8521097f69e615fea857ded4314a2c9aa5`。代码、已运行Notebook、摘要表、图和结论一并提供；原始/派生订单数据、逐订单预测、OOF和模型文件保留本地，可使用课程ZIP复现。本目录不是最终stacking交接或最终提交。

**审阅入口：[当前进展、结果和独立复现](docs/current_progress.md)。** 最新以`notebooks/feature_optimization.ipynb`及`versions/feature_optimization_v3/outputs/`为准。下文保留课程范围与早期实验过程；原主Notebook/交接明确标为历史版本。

**当前优化结论：** [特征优化Notebook](notebooks/feature_optimization.ipynb)及[中文说明](docs/feature_optimization.md)完成18组时间/路线/历史/样条比较，并补充成熟标签回测。新增方案未通过选择，保留上一轮Ridge和近期森林；三月MAE仍为6.847/6.913天。成熟历史CV恢复3,669笔评分订单，MAE为5.236/5.219天；旧4.54天仅是条件化口径。新结果在`versions/feature_optimization_v3/`。原主Notebook和正式交接仍为明确v1历史版本，方案冻结后需统一替换。最终测试未评分，stacking未拟合。共享范围仅为代码、已运行Notebook、摘要结果和结论；数据、逐订单预测、OOF与模型文件保留本地。

上一轮[baseline/变体复核](notebooks/baseline_variant_review.ipynb)及[中文解释](docs/baseline_variant_review.md)记录参数、输入log与近期权重的有限收益，结果在`versions/baseline_review_v2/`；本轮继续检查其特征和历史评价，不把两轮18组混为同一个网格。

**数据修正已落实：** 正式`prepare_data`默认使用筛查异常坐标、去重复参考点、邮编中位数；独立生成并核对的v2输入位于`versions/preparation_v2/data/eligible_orders.csv`。原始ZIP、旧`data/eligible_orders.csv`和模型/OOF/预测保留。以下首轮结果与审计过程保留历史语境，当前模型审查以本段链接为准。


首次整理：**2026-10-04**；更新：**2026-10-05**。模型路线已确认，本目录建立为独立的本地 Git 项目；先完成六步基础建模与输入交接，stacking 拟合仍为独立集成步骤。已核对 Phase 1 最终报告及教师反馈，并完成开发期六步：数据、baseline、有限改进、解释、报告素材及时间 OOF 交接。stacking 未拟合，最终测试未评分。材料先保存在本地，后续共享按明确授权执行。


**2026-10-05 数据基础复核：** [已运行审计 Notebook](notebooks/data_preparation_audit.ipynb) 独立核对原始7表。96,470笔有效订单、目标与非地理输入确认；旧邮编均值受异常坐标影响，已准备筛查后中位数候选输入。现有模型分数、OOF和pipeline仍基于旧地理处理；正式预处理已修正，下一轮用v2统一重训，当前交接暂不能视为定稿。审计没有训练/调参或测试评分，详见[新增核对结论](docs/validation_review.md#2026-10-05-补充数据基础复核)。

入口：[中文版导读](docs/中文版导读.md) · [已运行 Notebook](notebooks/delivery_regression_phase2.ipynb) · [英文报告素材与配图](docs/report_section.md) · [stacking 交接](docs/stacking_handoff.md) · [核对记录](docs/validation_review.md) · [反馈落实](docs/phase1_feedback_response.md) · [实验约定](config/problem_spec.json)。教师反馈支持保持当前目标和家族选择，重点补齐业务角色、论文特征到 Olist 的映射、指标理由、模型差异解释及图表证据。

## 1. Phase 2 要做出什么

Phase 1 已探索数据、关系与候选问题。Phase 2 要把选定问题变成可复现的预测实验：**明确目标与输入 → 准备数据 → 训练简单模型 → 验证改进是否有用 → 解释结果并选择最终模型**。

正式来源为 [AY2026/27 Sem1 Project Description V2](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html)，2026-10-04 重新读取；本地[老师原文截图 PDF](../01_当前小组项目/老师项目说明_原文截图.pdf#page=17)实际 pp17–18 对应 Phase 2 要求，p22 对应评分重点。下列页码均指 PDF 实际页码。

| 正式要求 | 应呈现的成果 | 来源 |
|---|---|---|
| 截止 **2026-10-11 23:59**，本阶段占项目 40% | 在截止前完成全组提交 | V2 Phase 2；PDF p17；当天 Canvas M2 |
| 全组 **1–2 个问题，覆盖 classification 与 regression** | 每个问题的目标、业务使用者和成功标准 | PDF p17 |
| 全项目合计 **2–3 个模型家族** | 有深度的比较；两个问题合理复用家族 | PDF p17 |
| 每个家族从最简单的 baseline 开始，再增加复杂度 | 说明复杂模型相对简单基线的收益；未提升时如实报告并保留简单方案 | PDF pp17–18 |
| 清理与特征工程 | 清理规则、目标计算、特征定义、预测时点及泄漏排除依据 | PDF p18 |
| 划分、交叉验证与调参 | 明确 train/test 或 train/validation/test，展示 CV 和调参方法；预处理在训练折内拟合 | PDF pp18、23 |
| 回归评价与模型选择 | MAE、RMSE、R²（按情境），比较表及最终选择理由 | PDF p18 |
| 可解释性与业务意义 | 重要特征或解释性分析、可执行洞察、限制；如用 stacking，说明其价值 | PDF pp18、22 |
| 全组 **6–8 页 PDF 技术报告** | 不计封面、参考文献、附录；报告包含 GitHub 链接和性能汇总表 | PDF pp17–18 |
| GitHub 上的 Notebook 和 Python 脚本 | 代码、环境依赖、运行顺序、随机种子与可复现说明 | PDF pp18、23–24 |
| 报告格式和 AI 声明 | 12pt、单倍行距，图注和表格标签统一，引用与 AI 使用声明完整 | PDF p25 |

[Canvas M2 提交页](https://canvas.nus.edu.sg/courses/94662/assignments/263614)在 2026-10-04 显示 **Due 11 Oct at 23:59、Points 40**，与 V2 一致。页面同时显示模块未发布，暂无法读取提交正文；显示的 Available until 18 Oct 不是正常截止日。若后续 Canvas 更新，以最新正式要求为准。

6–8 页是**全组报告**的预算，不是配送回归单独的篇幅。Phase 2 的重点是训练、评估和解释；应用部署属于 Phase 3。本阶段未要求重新单独完成一篇 literature review。

## 2. 模型顺序：正式要求与实施建议

老师没有规定固定的“一个 baseline 加一到两个 variants”。正式要求是家族数量、家族内简单基线以及有证据的复杂度提升。**Variant 指模型变体或改进版本；variance 指统计上的方差。** Stacking 在 V2 中是可选进阶集成，不是简单 baseline。

以下路线已于本轮确认，具体算法不是教师指定的唯一清单；最终须与全组 2–3 个家族的预算一致。

| 层次 | 拟用方法 | 目的 |
|---|---|---|
| 基础参照规则 | 直接预测承诺时长；训练集均值和中位数 | 判断学习模型是否比现有承诺或恒定预测更有用；这些参照不能替代家族内 baseline |
| 家族 A：线性 baseline | `LinearRegression` | 建立简单、可解释的拟合模型，观察多个特征共同预测的效果 |
| 家族 A：正则化变体 | `Ridge`（alpha 100） | 稀少月份导致历史折波动，有限正则化检查改善稳定性 |
| 家族 B：树 baseline | `DecisionTreeRegressor` | 建立单棵树的参照，探索阈值和非线性关系 |
| 家族 B：主要改进版本（Variant 1） | `RandomForestRegressor` | 与单棵树及线性模型公平比较，检验多棵树集成的收益 |
| 家族 C：后续集成模块 | `StackingRegressor`，组合已评估的 A/B 模型 | 检验组合预测能否带来额外改善；接收一致的数据和预测交接材料 |

所有模型共用目标、16个特征、数据划分和指标。已补充 Ridge alpha 10/100 检查，依据训练 CV 选100；随机森林试验的 log1p 目标没有优于原尺度方案，因此未采用。未实现额外稳健损失。

复杂模型必须通过结果证明价值。随机森林或 stacking 没有提升时，记录原因并选择更合适的简单模型，不预设 stacking 一定成为最终模型。

## 3. 配送回归问题与现有基础

**问题：在下单时，仅根据当时可获得的信息，预测该订单从购买到客户实际收货需要多少天。** 主要业务使用者为**客户服务/订单运营经理**，以订单时长估计辅助到货预期沟通；履约规划为次要用途。当前点预测不等于有覆盖率保证的到货区间，不直接宣称改善满意度或降低成本。

目标为订单级连续天数：

```text
lead_time_days = (实际收货时间 - 下单时间).总秒数 / 86400
```

这与 Phase 1 定义一致。预测的是完整配送时长，不是超过预计日期的迟到天数。直接读取课程7表，核对共享 v2 与 Phase 1 统一订单聚合，一笔订单一行；不使用 `.dt.days` 向下取整的共享字段作为连续目标。

Phase 1 已有 **96,470** 笔有效已送达订单，**306** 笔超过 60 天的有效订单全部保留。最大已观测距离、承诺时长和运费与目标存在探索性关联，支持继续研究，但尚不能说明模型能准确预测。来源：[配送回归 Notebook](../03_配送时长分析/03_delivery_lead_time_regression_eda.ipynb) cells 3、8–13、17、29–35；[数据与方法说明](../03_配送时长分析/数据与方法说明.md)。

Phase 1 的固定承诺规则历史 MAE 约 **12.73 天**，是全历史样本的描述性误差。Phase 2 必须在同一测试集上重新评价规则和模型，不能把 12.73 天直接当作未来测试的固定达标线。初步成功标准为：在预先确定的评价方案下，相对同样本的简单参照降低 MAE，并检查大误差、稳定性和业务适用性；尚不设无证据的数值提升目标。

### 允许和排除的输入

| 角色 | 初始范围 |
|---|---|
| 可评估的下单特征 | 原始承诺时长；卖家—客户距离代理及缺失标记；所有卖家是否同州/任一跨州；客户州；商品数、卖家数；商品金额、运费与比值；商品重量、品类；下单月份、星期等 |
| 目标与审计，不能作为输入 | 实际收货日期、配送时长及其副本、最终订单状态、迟到/按时结果、评论与评分；严格下单时点下不加入后续支付或审批信息 |
| 仅用于连接和审计 | `order_id`、`customer_id`、`customer_unique_id` 等标识，不将任意 ID 当作数值预测变量 |

Phase 1 导出表含结果与审计字段，不能整表直接作为 X。多卖家订单的州关系使用全部卖家的关系，不能用首个卖家州代表全单。承诺字段需假设是下单时原始报价，历史归档不能证明其从未被修改。来源：Notebook 00 cells 42–45；Notebook 03 cells 12–13。

### 已实施的评价方案

固定训练：购买且收货早于2018-03-01，53,644笔；验证：3月购买、标签7月前可见，7,003笔；4–6月为标签成熟间隔；锁定测试：7月起12,507笔。3,673笔早期未收货订单另计，未当脏数据永久删除。日期根据时间覆盖、在首次模型评价前确定；老师没有规定这些日期。训练CV为3个扩展时间折，具体见 `outputs/tables/fold_summary.csv`。CV预测样本受3月截止标签筛选，最近折会缺少极慢订单，因此不是未截尾的未来效果估计。

每个验证起点的训练订单还须已实际送达，保证当时能看到训练目标。缺失填补、编码、缩放和任何目标变换均只在相应训练折拟合。所有模型使用相同评价订单；测试结果用于最终评价，不用于选特征、调参或决定长尾处理。

**MAE 为主要选择指标**，RMSE 和 R² 为补充；同时查看平均高估/低估、绝对误差高分位、长尾和主要路线切片。所有预测统一换回“天”评价。有效长尾保留，训练与评估策略避免只追求总体平均而忽略极慢订单。

## 4. 基础建模模块的工作与产出

六步已完成开发期产出；最终测试评价留待基础与 stacking 设计冻结后。每步实际位置如下。

| 顺序 | 要完成的工作 | 对应产出与完成标准 |
|---|---|---|
| 1. 固定问题与实验约定 | 确认下单预测、连续目标、完成配送样本范围、业务使用者、指标及模型家族；核对最新共享数据构建并建立文献—特征映射 | `config/problem_spec.json`、`config/feature_roles.csv`、`docs/literature_to_features.md`，来源指纹记录于 `outputs/tables/source_inventory.csv` |
| 2. 准备数据和划分 | 检查订单唯一性、日期/目标有效性、缺失和多卖家聚合；固定时间划分与各 CV 折 | `src/delivery_regression.py`、`data/split_manifest.csv`、`data/cv_fold_manifest.csv`、`outputs/data_quality.json`；预处理仅在训练折拟合 |
| 3. 运行参照与简单 baseline | 计算承诺、训练均值/中位数参照；训练线性回归和单棵决策树 | `outputs/tables/model_comparison.csv`、`models/*_development.joblib`、`outputs/tables/cv_results.csv`；相同验证订单比较 |
| 4. 完成 Variant 1 | 对随机森林做有限、可解释的调参；按需要验证长尾处理 | `outputs/selected_models.json`、`outputs/tables/cv_summary.csv`；配置由训练CV选，不用最终测试调参 |
| 5. 解释与整理报告素材 | 以开发期 CV/验证结果分析模型差异、误差、重要特征、长尾与业务限制；基础和 stacking 方案固定后共同评价锁定测试集 | `outputs/figures/`、`outputs/tables/`、`docs/report_section.md`；明确开发结果，未最终测试 |
| 6. 交接 stacking 输入 | 统一折编号与订单，生成基模型 OOF 预测并提供运行方法 | 4个pipeline、`outputs/oof/base_oof_predictions.csv`、`docs/stacking_handoff.md`；39,445笔共同OOF，不含验证/测试 |

代码入口为 `notebooks/delivery_regression_phase2.ipynb`，配套 `src/delivery_regression.py` 整理数据与实验逻辑。交付中同时记录依赖版本、随机种子、数据路径、运行顺序和 AI 使用说明。数据、模型文件及逐订单预测先本地保存，不默认上传。

报告素材需要说明问题、数据和特征、划分/CV、模型与调参、比较结果、解释性、业务用途和限制。正文篇幅由全组 6–8 页预算统筹；细节可放附录和 Notebook，不重复整篇 Phase 1 EDA。

## 5. Stacking 的交接边界

基础建模模块完成前述参照、线性/树 baseline、随机森林及其比较。后续集成模块训练 stacking、评价其额外收益，并参与最终选择；两个模块首先共用数据、目标、划分与指标。

交接至少包含：数据版本/指纹、订单 ID 与顺序、特征白名单、目标定义、训练/验证/测试时期、CV 折、各模型 pipeline/参数/环境，以及各基模型的 **OOF（out-of-fold）预测**。

OOF 的意思是：某笔训练订单的预测，由**没有用这笔订单训练过**的模型生成，供 stacking 学习如何组合基模型。不能用模型对自己训练样本的拟合预测替代 OOF，也不能用测试标签训练组合器。时间顺序 OOF 的最早一段没有更早训练数据，应统一标为预热期并排除相应元模型训练行；不能编造预测。具体折设计在实施前固定，不能直接假定默认 stacking CV 满足时间预测要求。

测试预测与 OOF 分开命名、保存；最终测试在模型方案固定后统一进行。较早的基础模型交接优先提供开发期结果和 OOF，避免后续根据测试分数反复调整 stacking。

## 6. 当前核对记录与下一步

- **官方要求**：2026-10-04 读取 V2；核对本地 PDF 实际 pp14–18、22–25。PDF p15 的配送回归示例与本问题一致；p16 五项 Problem Scoping Checklist 继续适用。
- **Canvas**：当天 M2 Due 与 V2 一致；提交正文因未发布模块暂不可见，不将此状态推断为延期。
- **GitHub**：当天读取最新 `main`，提交 `ce57de2ef86ce3e8d887c7703326b1139f95070b`；公开文件树包含 Phase 1 Notebook 和成果，未见 Phase 2 建模文件。配送 `03` 与已发布版本内容相同，路径改为 `notebooks/phase1/`；00 的方法代码未变，月订单量文字有修正。
- **本地副本**：已 `git fetch` 获取远程引用；工作树仍是 `83ad035`，尚未切换或 pull 到新目录布局。独立工程已读取远程提交对应00源码核对方法，不改动原共享工作树。
- **共享基础**：[最新 00](https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project/blob/ce57de2ef86ce3e8d887c7703326b1139f95070b/notebooks/phase1/00_dataset_overview_eda.ipynb) cells 38–46；cell 45 提供订单聚合。源数据仍用课程固定版本，不另下载替代数据。
- **开发结果**：已运行9个代码cell，无错误；Ridge验证MAE6.91天，随机森林7.01天，单树7.20天；这些是相同7,003笔3月订单上的开发结果。
- **2026-10-05 反馈补充**：最终 Phase 1 报告共19页，配送回归正文 p8、Figure 6 p14。读取教师反馈后保留问题与模型路线，补充业务使用、文献映射、指标选择理由与结果证据要求。原报告和反馈归档到 `01_当前小组项目/Phase1_提交与反馈/`，不修改原文。
- **独立 Git 项目**：在本目录初始化 `main`，跟踪方案、问题配置和后续实现；原始数据、逐订单预测和拟合文件排除。当前没有远程地址，没有上传。

下一步交接 stacking，冻结方案后共同重新拟合与评价锁定测试。森林未胜过3月线性；学习模型仍平均低估约4.3天，37笔>60天验证订单MAE约53天，不能宣称已能可靠预测极慢配送。历史要求核对与反馈核对日期分别保留。

## 7. 本地复现

需要 Python 3.12。全新目录中创建环境并安装固定依赖（已建立的环境可以跳过前两步）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

设置 `OLIST_CSV_DIR` 为课程CSV目录，或设置 `OLIST_ARCHIVE` 为课程ZIP绝对路径；两者都未设置时读取相邻课程配套资料目录。只读取来源，不修改原件。

```bash
.venv/bin/python scripts/run_notebook.py
.venv/bin/python scripts/verify_outputs.py
```

本轮独立模型审查（使用已生成的审计v2数据）：

```bash
.venv/bin/python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb
.venv/bin/python scripts/verify_baseline_review.py
```

本轮特征优化与成熟标签回测：

```bash
.venv/bin/python scripts/run_notebook.py notebooks/feature_optimization.ipynb
.venv/bin/python scripts/verify_feature_optimization.py
```

特征Notebook默认从头复现18组配置；已运行4个代码cell，并独立核对240个历史时点。`mature_backtest`预测含43,114笔历史评分订单；原开发训练OOF另保留39,445笔，不能合并混用。历史/样条候选没有被采用，当前保留输入仍为16字段。

`verify_outputs.py`仍核对首轮正式交接；新审查使用独立核对脚本，不把两版本混在一起。

也可在Notebook中从头运行，或用 `.venv/bin/python src/delivery_regression.py` 运行核心实验（后者不更新Notebook显示输出）。数据、pipeline、逐单预测和OOF会写入本地被忽略目录；简明表/图、环境、代码和已运行Notebook由本地Git跟踪。`scripts/build_notebook.py`是Notebook结构生成工具，会清除旧输出，正常复现不需要先运行它。

当前9个代码cell和独立核对通过。核对脚本重新计算指标、检查时间/字段边界、重载模型以及与Phase 1/原始表对齐，不训练stacking或评价测试。主要运行软件版本随依赖和结果记录；不保证不同操作系统的耗时或浮点末位完全相同。

AI协助实现、解释和调试；来源、运行证据和限制已按审查流程核对，提交责任仍由项目团队承担。

[Phase 1 配送分析](../03_配送时长分析/README.md) · [当前小组项目](../01_当前小组项目/README.md) · [课程项目指南](../../01_课程信息/03_作业与项目指南.md)


复现首轮历史数字时，应显式调用 `prepare_data(geography_policy="legacy_mean")` 后运行相同训练流程；默认 `prepare_data()` 现在产生已审查的v2地理输入。`notebooks/data_preparation_audit.ipynb`只审计和生成独立v2，不拟合模型；不要把运行该审计与更新模型结果混为一谈。现有主Notebook数值是v1历史输出，后续重跑会得到v2实验，需要同步改写数值结论。
