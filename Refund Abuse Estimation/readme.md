This project is similar to my experience during internship at Apple as a data analyst. Due to privacy of the data of the company, I reproduce one that is similar in process and ideas with the virtual, man-made dataset customer_churn_data.
This project is about estimating whether users will abuse the refund policy of a company, which is an integration of machine learning techniques, including:
- Preparations for modeling:
  - Datatype adjustment according to the needs.
  - Checking and dealing with null values.
  - Feature engineering
  - Upsampling
  - Applying PCA to reduce colinearity in predictors
- Clustering the data
  - Using K-Means
- Building the model:
  - Three models are applied: logistic regression, support vector machine and random forest
- Evaluation of the model:
  - Variable importance
  - ROC and its AUC
  - Precision-Recall
