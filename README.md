# Customer Churn Prediction — Generative AI for Business Applications

This analysis presents a machine learning approach for predicting customer churn in a banking
dataset. The main objective is to determine whether a customer is an Existing Customer or an
Attrited Customer based on demographic, account and transaction-related characteristics. The
prediction can help a bank identify customers who may leave and take suitable retention actions.
Because failing to identify a customer who actually churns can be costly, recall is given
particular importance in the model evaluation.
The methodology starts with understanding and preparing the customer dataset. CLIENTNUM,
which is only a unique customer identifier, is removed. The target variable is converted into
binary values, with Existing Customer represented by 0 and Attrited Customer by 1. Categorical
variables are converted into numerical representations using hierarchical mapping, dummy
encoding and label encoding. The data is subsequently divided into 70% training and 30%
testing data using stratified sampling.
Three machine learning algorithms are then applied: Decision Tree, Random Forest and Gradient
Boosting. Their performance is compared using accuracy, precision, recall and F1-score. The
Decision Tree obtained approximately 93% test accuracy, but the difference between its perfect
training performance and lower test performance indicates overfitting.
The ensemble models produced better results. Random Forest and Gradient Boosting both
achieved approximately 96% test accuracy. Gradient Boosting achieved approximately 93%
precision, 81% recall and 86% F1-score, giving it the best overall balance in the reported results.
The feature-importance analysis indicates that transaction count, transaction amount, revolving
balance and relationship count are among the most important predictors of churn. In particular,
Total_Trans_Ct had the highest importance at 0.33. Thus, the study demonstrates that customer
behaviour and account activity can be effectively used to predict attrition and support customer
retention planning.
