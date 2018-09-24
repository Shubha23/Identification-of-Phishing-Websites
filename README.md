# Phishing-Website-Classification
Application of Machine learning and Feature selection technqiue for classification of phishing websites
**************************************************************************************************************************
Project goal -
The objective of this project is to classify phishing and legitimate websites using prominent machine learning algorithms.
Apply feature selection and compare the models' performances without feature elimination.
***************************************************************************************************************************
Techniques used -
For classification - Random forest Classifier, SVM Linear and Logistic Regression from Scikit-learn
For feature selection - Gain Ratio attribute eval using Ranker's method from Weka-3.8 software GUI 
***************************************************************************************************************************
Dataset - 
Source - UCI Machine Learning Repository, published in 2015. 
It contains thirty attributes related to each website 
There are 11056 total instances (websites). 
The dataset features could be categorized into four major categories -
1. Address bar based attributes (12 features), 
2. Abnormal Based features (6 features), 
3. HTML and Java Script based features (5 features)
4. Domain based features (7 features).
***************************************************************************************************************************
Performance metrics - For each model
- ROC plots' area under the curve values
- Accuracy score
- Confusion matrix
***************************************************************************************************************************
Conclusions -
- Random Forest outperforms the other two.
- Performance remains unchanged on removal of lowest three features as per their gain ratios.  
****************************************************************************************************************************
To compile and run -
python Classification.py
***************************************************************************************************************************
*Note - Update file path to local directory pathname.

******************************************************* END OF FILE ********************************************************
