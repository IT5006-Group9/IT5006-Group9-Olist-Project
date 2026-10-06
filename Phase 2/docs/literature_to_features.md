# 文献如何落实到配送回归

2026-10-05 核对出版方摘要。摘要可支持以下问题/方法和字段类别；未获得全文的特征细节不标成已完整复现。精确 Olist 公式见 [feature_roles.csv](../config/feature_roles.csv)。

| 来源与已核对依据 | 对本问题的启示 | Olist 实现与限制 |
|---|---|---|
| Salari et al. (2022)，INFORMS 摘要：在 JD.com 场景估计配送时长分布，并据不对称成本选择承诺；扩展 quantile regression forest | 时长估计与承诺决策有关，但点估计、分布和成本优化是不同输出 | 以 `orders.order_estimated_delivery_date - order_purchase_timestamp` 得到已知承诺，作为输入及固定参照；回归目标仍是实际时长。本模块不复制分布预测或宣称优化销售 |
| L. Zhang et al. (2023)，Springer 摘要：DGM-DTE 用商家、发/收地址、支付时间估计时长，兼顾常见与稀少时长 | 地理、商家和时间信息值得研究；改善尾部可能牺牲总体表现，需要同时评价 | 客户/卖家邮编连接地理中心，按明细计算距离并取订单最大值；全部卖家州关系、卖家数为可解释代理。购买时间生成日历特征，不能把它说成等同于论文支付时间 |
| 同一 DGM-DTE 摘要中的支付时间 | 论文特征必须按具体预测时点重新审查 | 严格下单时点排除之后的支付/审批；不直接照搬论文全部输入，不用任意商家 ID 当数值 |
| Phase 1 配送 EDA（最终报告 p8/Figure 6 p14） | Olist 内承诺、距离、运费已有探索信号；相关性不能代替预测结果 | 将订单金额、运费、比值、商品属性和商品数纳入初始白名单；这些主要是本地业务/EDA 驱动，摘要未明确列出它们时不虚构论文来源 |

实施选择是线性/树基线、有限正则化检查与随机森林，而非复现图网络或分位数承诺系统。模型增加复杂度的价值由相同验证订单上的误差、时间折稳定性和尾部表现判断；论文在别的数据上的分数不能直接作为 Olist 达标线。

参考文献：

- Salari, N., Liu, S., & Shen, Z.-J. M. (2022). Real-time delivery time forecasting and promising in online retailing: When will your package arrive? *Manufacturing & Service Operations Management, 24*(3), 1421–1436. [出版方摘要](https://pubsonline.informs.org/doi/10.1287/msom.2022.1081)。
- Zhang, L., Wang, M., Zhou, X., Wu, X., Cao, Y., Xu, Y., Cui, L., & Shen, Z. (2023). Dual graph multitask framework for imbalanced delivery time estimation. *DASFAA 2023*, 606–618. [出版方摘要](https://link.springer.com/chapter/10.1007/978-3-031-30678-5_46)。

技术方法来源：[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 的抽样树平均与参数说明；[StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) 的交叉验证预测与默认 KFold 说明。实际运行版本固定为 requirements.txt 中的 scikit-learn 1.7.2；时间 OOF 在本模块显式生成，不能直接将默认 stacking CV 当作时间预测验证。
