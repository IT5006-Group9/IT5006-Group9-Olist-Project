
# Literature Review

## Delivery Performance and Business Outcomes

Faster delivery promises can increase short-term sales, profits and order value, but overly aggressive promises may increase the risk of late delivery relative to customer expectations and reduce retention (Cui et al., 2024). Actual delivery performance also influences post-purchase outcomes: late deliveries are associated with lower review ratings and satisfaction, while improved logistics performance can increase purchase probability and seller sales (Ravula, 2022; Vakulenko et al., 2019; Deshpande & Pendem, 2023). These findings highlight a key trade-off for e-commerce platforms between offering attractive delivery promises and maintaining reliable fulfillment.

## Modelling Approaches for Delivery Prediction

Delivery performance can be analyzed through both regression and classification. Regression models estimate continuous outcomes such as delivery lead time, whereas classification models identify high-risk outcomes such as late or failed delivery. Kandula et al. compared Random Forest, XGBoost, LogitBoost, artificial neural networks and Decision Trees for delivery-success prediction and showed that predictive outputs could support delivery planning and reduce failed delivery attempts and logistics costs (Kandula et al., 2021).
For structured transactional datasets such as Olist, tree-based models are particularly relevant because they can capture nonlinear relationships and feature interactions and generally perform strongly on medium-sized tabular data (Grinsztajn et al., 2022). Logistic regression can serve as an interpretable baseline, while Random Forest and XGBoost provide more flexible alternatives for both delivery-time regression and late-delivery classification.

## Methodological Considerations

Class imbalance is an important consideration when predicting late delivery because late orders may represent a smaller proportion of observations. Although techniques such as oversampling, undersampling and SMOTE are commonly used, they should not be applied automatically. Van den Goorbergh et al. found that imbalance corrections did not necessarily improve discrimination and could result in poorly calibrated predicted probabilities (van den Goorbergh et al., 2022). In some cases, classification-threshold adjustment may provide similar improvements in sensitivity without altering the training distribution.
Data leakage is another major concern in delivery prediction. Leakage can produce overly optimistic and poorly reproducible model performance when information unavailable at prediction time is included in model development (Kapoor & Narayanan, 2023). For late-delivery prediction, variables that only become available after delivery completion should therefore be excluded from predictors. For example, the actual delivery date may be used to construct the late-delivery target but should not be used as an input feature. Preprocessing, feature selection and any resampling procedures should also be conducted using training data only (Kapoor & Narayanan, 2023).

## Feature Engineering and Evaluation

Useful predictors of delivery performance may include temporal, transactional, product, freight and geographic variables. Kandula et al. showed that both order-level and location-level information can contribute to delivery prediction (Kandula et al., 2021). For Olist, candidate features therefore include purchase timing, product characteristics, freight-related measures, seller and customer locations, and geographic separation.
Evaluation should reflect the prediction task. For imbalanced late-delivery classification, accuracy alone may be misleading, so precision, recall, F1-score and AUROC are more informative (Kandula et al., 2021). Calibration may also be relevant when predicted probabilities are intended to represent delivery risk (van den Goorbergh et al., 2022). For delivery-time regression, MAE, RMSE and R² can be used to assess prediction error and explained variation.

## Research Gap and Relevance to Olist

Existing research links delivery promises and actual delivery performance to customer satisfaction, retention and commercial outcomes (Cui et al., 2024; Ravula, 2022). This creates an early risk identification problem for e-commerce platforms and sellers: can orders at high risk of late delivery be identified before delivery completion using information already available during the order or fulfillment process?
The Olist dataset provides an opportunity to examine this question using temporal, freight, product, transactional and geographic features. High-risk orders could be prioritized for closer monitoring or fulfillment follow-up, supporting more proactive action before a delivery promise is missed. Delivery-time regression can complement classification by estimating expected delivery duration, providing additional support for delivery planning and customer expectation management.




## References

1.	Cui, R., Lu, Z., Sun, T., & Golden, J. M. (2024). Sooner or later? Promising delivery speed in online retail. Manufacturing & Service Operations Management, 26(1), 233–251.
2.	Ravula, P. (2022). Impact of delivery performance on online review ratings: The role of temporal distance of ratings. Journal of Marketing Analytics, 11(2), 149–159.
3.	Vakulenko, Y., Shams, P., Hellström, D., & Hjort, K. (2019). Online retail experience and customer satisfaction: The mediating role of last mile delivery. The International Review of Retail, Distribution and Consumer Research, 29(3), 306–320.
4.	Deshpande, V., & Pendem, P. K. (2023). Logistics performance, ratings, and its impact on customer purchasing behavior and sales in e-commerce platforms. Manufacturing & Service Operations Management, 25(3), 827–845.
5.	Kandula, S., Krishnamoorthy, S., & Roy, D. (2021). A prescriptive analytics framework for efficient e-commerce order delivery. Decision Support Systems, 147, 113584.
6.	Van den Goorbergh, R., Van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression. Journal of the American Medical Informatics Association, 29(9), 1525–1534.
7.	Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9).
8.	Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35, 507–520.