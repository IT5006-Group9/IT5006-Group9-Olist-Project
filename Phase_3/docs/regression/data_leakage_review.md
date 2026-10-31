# Regression data leakage 核查 — 2026-10-31

## 结论与范围

本次核查基于小组仓库提交 `8e66a6d`，覆盖当前 HGB_risk 主流程、初始三模型 stack，以及后来新增的 14 个 Phase 2 组合的时间迁移和补充测试路径。**在已核对的特征、训练标签、验证标签、预处理和风险／stacking 边界中，没有发现需要通过重新训练修复的直接目标或未来标签泄漏。** 这不等于从现有归档数据可以证明每个输入在真实购买当时均已存在，也不等于所有评价偏差均已消除。

本轮没有调整训练规则、特征、模型、参数、分数或原始评估记录，没有重新拟合模型。审计脚本仅读取原始数据和已有缓存，在独立 JSON 中保存结果。当前检查记录见 [checks.json](../../outputs/regression/leakage_review_20261031/checks.json)。

## 已核实的边界

### 1. 目标与购买时输入分离

预测时点为 `purchase_ts`；目标为 `(delivered_ts - purchase_ts)` 的小数天数。最终状态只用于界定有可观察结果的完成配送总体，不作为预测输入。主预处理器明确选择购买／报价、订单商品、卖家、商品属性、地理和日历列，丢弃其余列；订单 ID、客户唯一 ID、审计异常标记不进入模型。

- 独立从课程原始订单重算了全部 **96,203** 笔建模订单的目标及购买／收货时间，与已运行输入一致。
- 对原始订单的实际收货、carrier、approval 时间及最终状态进行整体扰动后，全部 **98,666** 笔购买时特征保持完全一致。
- 在 **160** 笔确定性抽样测试订单上，删除或扰动真实目标、收货、最终状态等字段，最终模型预测保持完全一致。这一推理检查是抽样检查，不声称逐笔重放了全部测试预测。

### 2. Gap 与标签可用性同时约束训练

每次模型拟合均要求：

```text
purchase_ts >= 2017-01-01
purchase_ts < fitting_cutoff - 39 days
delivered_ts < fitting_cutoff
```

39 天来自在 2018-01-01 前已经收货的 2017 购买订单，不使用 2018 测试标签估计 gap。等于截止时间的收货标签也不能用于该次训练。仅设置 gap 仍可能保留超慢且尚未收货的订单，因此最后一条是必要条件。

| 拟合窗口 | 训练数 | 评价数 | Gap 后仍待收货、被额外剔除的训练订单 | 实际进入训练的未知标签 |
|---|---:|---:|---:|---:|
| January validation | 34,246 | 7,068 | 43 | 0 |
| February validation | 42,450 | 6,551 | 62 | 0 |
| March validation | 47,940 | 6,996 | 81 | 0 |
| April validation | 54,882 | 6,715 | 103 | 0 |
| May validation | 62,130 | 4,406 | 133 | 0 |
| Final June–August assessment | 68,916 | 18,603 | 52 | 0 |

表中数量通过原始数据重建的布尔条件与实际函数输出逐一比较，不仅依赖保存表格中的 `0`。每折训练和评价订单 ID 互斥；June–August 测试订单不进入 January–May 开发训练或验证。

### 3. 验证标签必须在选模型之前可知

January–May 的模型选择仅评价 `delivered_ts < 2018-06-01` 的标签，核对后未知验证标签为 **0**。验证订单可以在其购买月份之后收货，只要结果在 6 月 1 日选择模型前已经可知。

“标签发生在预测之后”本身不是泄漏：配送结果本来就要发生在购买预测之后。必须防止的是把在**训练／选模型截止时仍未知**的结果，提前用于拟合或选择。

测试按 June–August **购买时间**纳入，不能再用“必须在 August 内收货”排掉慢单；测试采用归档中可观察的完成配送结果，评价仍受总体完整性限制。

### 4. 预处理与 slow-order risk

- 独立检查五份实际保存的预处理器：填补中位数、编码类别字典及拟合订单指纹均与各折训练输入匹配；并未在全数据上拟合后再切分。
- 重建五个外层训练分区与最终训练分区内的 **63 个已拟合月度风险窗口**，核对训练 ID、数量、标签时间及预测订单不重叠。
- 风险目标是历史订单是否超过 30 天；每笔回归训练订单使用过去月份预测出的风险概率，不使用该订单自己的慢单答案。January–March 2017 为明确的冷启动块：0 概率加不可用标记。
- 最终测试的风险概率由最终训练分区的辅助模型计算，不读取测试订单真实时长；Phase 2 的随机 OOF 概率未复用。

### 5. 新增 stacking 路径

- 重新构造 14 组合实验的 **5 个外层 + 15 个内层窗口**，其订单指纹与保存的 `time_audit.csv` 一致。内层基模型的训练标签必须在内层拟合时已知；用于外层权重学习的 meta 标签必须在外层截止前已知。
- 补充最终测试使用的 **3 个内层 meta 窗口**也按实际训练总体重建，核对数量与订单指纹，均满足截止限制。
- 权重来自前向 OOF；没有用测试误差拟合权重。选出的 stack／mean 由验证表确定，而非取测试分数最小的方案。
- 新增 stacking 边界测试共 **11 项**通过，涵盖截止等号、gap、未知 meta 标签、测试月份隔离及测试结果列删除。
- 本轮没有重跑 14 组合的所有模型或重放其全部私有拟合矩阵。该部分的额外支撑是已检查的源码、独立重建的实际窗口和此前保存的 [组合验证记录](../../outputs/regression_stacking_chronological/verification.json)。

### 6. 部署路径与冻结配置

主流程的数据、特征、切分、学习、风险和评分源码与原始冻结指纹一致；冻结后的变化仅为报告生成与检查模块。部署导出器校验冻结描述、训练总体和输入指纹，另存固定配置复现模型。应用将购买输入与 reveal-only 真实结果分开，预测函数只接收购买输入。

部署源代码及 manifest 已核读；本次没有重新部署或重新拟合部署模型。参见 [部署交接](deployment_handoff.md) 和 [部署 manifest](../../deployment/assets/manifest.json)。

## 最终报告必须保留的限制

| 限制 | 性质与影响 | 准确表述及处理 |
|---|---|---|
| 归档报价、商品／订单明细、卖家属性和静态地理资料缺少版本／变更时间日志 | 无法验证的 as-of 可用性假设，不是本次已证实的直接泄漏；不能证明归档值从未被事后修改 | 声明假定这些字段在购买时可用；不写“所有字段历史可用性均被证明” |
| May 标签仅成熟 65.3% | 严格排除未来标签后的选择偏差；不能把“零未知标签”说成“零选择偏差” | May 4,406/6,749；模型排名对其敏感；保留已保存的成熟度诊断，不用其事后更换模型 |
| 总体仅含归档中已完成配送订单 | 排除了取消／未完成订单，且没有权威抓取截止时间；限制对所有购买订单的外推 | 明确条件总体；不声称预测覆盖所有下单结果，也不声称归档尾部完全无截尾 |
| 原始三模型 stack 的基模型参数先在主 CV 选定 | 不是完整嵌套超参数评价，CV 只是开发证据；不等同于测试标签直接进入拟合 | 原始 stack 与后来的 14 组合内层选择分开描述 |
| June–August 测试已有历史曝光 | Phase 2 以及后续补充评估已接触这些订单，新测试锁不能消除历史曝光 | 使用“retrospective chronological assessment”；明确追加 stacking 测试并未重新选择主模型；不把全项目说成“测试仅用过一次” |

未来如重新设计实验，应事先固定较成熟验证月份、观察期限和选择标准，再使用新的独立时段确认。当前审查没有发现必须以修复直接泄漏为由重新训练的证据；这些限制也不能靠重跑相同数据自动消除。

## 复核方法

使用回归已验证环境、固定课程 ZIP，以及此前完成的本地回归数据／模型缓存。从仓库根目录运行：

```bash
python Phase_3/scripts/review_regression_leakage.py \
  --archive /absolute/path/IT5006_Project-Data.zip \
  --output /tmp/regression-leakage-review-new.json
python -m unittest discover -s Phase_3/tests -p 'test_stacking_chronological.py' -v
python -m unittest discover -s Phase_3/tests -p 'test_selected_stacking_holdout.py' -v
```

私有缓存不在 Git 中；需要时可使用 `--private-root`、`--models-root` 指向已有运行目录。没有缓存时先按[原回归复现说明](../../src/regression/README.md)准备完整运行，不能把缺少缓存当成审计已经通过。脚本不拟合模型、不改原有 audit／lock／成绩，拒绝覆盖已存在的复核 JSON。

## 报告可用英文表述

> A final audit found no direct target or future-label leakage in the reviewed regression paths. Training required both purchase before the forecast origin minus the embargo and receipt before the origin; selection used only validation outcomes known before 1 June. Preprocessing and auxiliary risk were fitted within their training partitions. This does not establish the historical availability of every archived metadata field, remove completed-delivery or label-maturity selection, or turn previously exposed historical test orders into a new independent holdout.

正式正文见 [final_report_regression.md](final_report_regression.md)。核查代码与说明由 Codex 辅助整理；该检查是项目内部验证，不是独立外部审计。
