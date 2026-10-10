# Delivery Lead-time Regression · 当前最佳方案

冻结方案：**HistGradientBoostingRegressor + 购买信息、时间与预测风险特征**（leaves63__time_risk）。这是本轮有限搜索中的训练CV最优，保留原数据和分区；不声明全局最优。共享日期：2026-10-10。

## 先看这些

- [已运行 Notebook](notebooks/best_scheme_review.ipynb)：目标、数据／特征、防泄漏、参数比较、最终结果及慢单表现。
- [完整参数与选择](config/selected_model.json)、[结果摘要](results/result_summary.json)、[参数比较表](results/tables/parameter_comparison.csv)。
- [测试结果表](results/tables/test_metrics.csv)、[慢单结果表](results/tables/test_duration_groups.csv)、[五折配对变化](results/tables/paired_fold_changes.csv)。
- [原实验 CV 核对](results/cv_verification.json)、[原实验模型核对](results/verification.json)、[分享包核对](publication_validation.json)。

## 结果

| 指标 | 原最佳参数 | 本方案 |
|---|---:|---:|
| 训练五折平均MAE | 4.2729天 | **4.2378天** |
| 同31,836订单测试MAE | 4.1923天 | **4.1505天** |
| 测试RMSE | 7.5028天 | **7.4651天** |
| 测试R² | 0.3611 | **0.3675** |
| 误差≤3天占比 | 56.41% | **56.93%** |

最大叶节点由31增至63；loss=absolute_error、max_iter=300、learning_rate=0.05、min_samples_leaf=30、l2_regularization=10、early_stopping=False、random_state=33。8固定设置在time_risk上比较，随后以选中参数做time-only简化检查；按完整五折平均MAE选定后才进行本轮测试。五折均小幅改善，测试MAE降低0.0418天／约1.00%，相当于约60分钟。30–60天订单MAE18.8525天，>60天66.1881天，长尾问题仍然明显。

固定课程ZIP与screened_median地理V2；96,470笔完成配送有效目标，306笔>60天保留。随机67/33、训练64,634／测试31,836、seed33、训练5折。输出为购买到客户收货的连续天数，输入只使用购买信息；归档报价／卖家／商品属性可用性是明确假设。新增风险是对原购买输入的学习变换，外层回归训练内部3折交叉拟合，验证风险只由外层训练生成；最终拟合使用训练5折风险OOF和全训练推理分类器。

日期特征主要反映同一历史混合总体，不能据此宣传未来月份表现。随机测试与开发折此前已多次用于探索，因此本次为后续开发比较，不是全新独立确认；CV选定后不按测试结果切换模型。风险输入的额外CV收益约0.0021天，不能称显著改善。

## 报告证据补充（2026-10-10）

[报告证据 Notebook](notebooks/report_supplement_review.ipynb) · [表格、图与口径说明](docs/report_evidence_guide.md) · [完整 Train–CV–Test 表](results/report_supplement/tables/train_cv_test_comparison.csv)。补齐15行同随机切分的参照／baseline／variant、最终模型训练成绩、HGB分组置换重要性、误差图、分组样本数和时间矛盾的评价敏感性。最终模型／参数／测试成绩不变。

最新 stacking 结果另列：外层CV4.2374 vs HGB4.2378天，约36秒收益；未拟合最终stack或评分测试，不填造Train/Test。共享报告由整合流程更新，本包仅补充工程证据。

固定模型的完整重建命令与 `--with-cv` 说明见报告证据指南。数据、逐单预测和模型仍不上传。

## 重建冻结的最佳模型

进入本文件所在目录。Python3.12环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_bundle.py
.venv/bin/python scripts/run_notebook.py notebooks/best_scheme_review.ipynb
```

仅重建／检查课程数据和分区，不训练：

```bash
.venv/bin/python scripts/reproduce_best.py --archive /absolute/path/IT5006_Project-Data.zip --output runs/prepare_check --prepare-only
```

训练并评价已选定模型，生成5折风险OOF、推理模型和测试结果：

```bash
.venv/bin/python scripts/reproduce_best.py --archive /absolute/path/IT5006_Project-Data.zip --output runs/best_reproduction
```

如需同时重建最佳模型的5折回归CV／OOF，在上述命令追加`--with-cv`，外层训练内将重新生成3折风险；该模式计算量更大。均要求全新输出目录；产出保存在被忽略的runs/，不训练Stacking。划分／输入／扩展购买特征的哈希必须与原实验一致。默认仅重建已冻结冠军，不重复所有参数候选或历史方案。源码保留共用类与原搜索模块；公开复现入口为reproduce_best.py，旧完整搜索函数依赖历史本地缓存，不是此分享包的运行入口。

数据、模型权重、逐单预测和OOF由课程数据在本地生成，不包含在Git分享包中。共享Notebook使用原实验指标／图并重新执行，便携修改仅涉及目录入口和不可共享历史文件的哈希核对；没有改变拟合参数或结果。源与结果指纹见[manifest](bundle_manifest.json)和[来源](publication_provenance.json)。分享包已从ZIP独立重建输入、分区与购买特征，并用原最佳模型重新加载验证全部测试指标；未再次完整训练所有参数候选。

此次是独立共享快照，原历史成果保持；没有拟合／更新Stacking，也没有替换正式交接或Google Docs。AI辅助实现与解释；结果依据实际运行、保存记录与核对。
