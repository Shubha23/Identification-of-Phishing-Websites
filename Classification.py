'''
# Aim - Classification of phishing and benign websites.
# Recursive feature elimination (RFE) is applied to remove lowest 3 features as per their gain ratio (see note below)
# Algorithms applied - Random Forest Classifier, Logistic Regression and SVM
# Performance metrics - For each model, generates ROC plots, accuracy_score and confusion matrix.
*Note - GainRatioAttribute eval using Ranker's method was applied in Weka-3.8 software GUI to achieve 
        ranking of all features based on their gain ratio. 
        After applying brute-force method, it was found that eliminating lowest 3 features maintain the
        performance of classifiers at the same level, thereby showing redundancy of features in original set.
'''
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,roc_curve,auc, confusion_matrix
from sklearn import preprocessing

# Read the data
data = pd.read_csv('./Dataset.csv')

# View first and last 5 observations
print(data.head())
print(data.tail())

# Describe statistical information of data
print(data.describe())
# Check column types
print(data.info())                # All comumns are int type, so no change is required

# Look for missing values
print(data.isnull().sum())        # No missing values found, so no need to drop or replace any value

# Generate correlation matrix
print(data.corr())

# Prepare data for models
X = np.array(data)[:,:30]
y = np.array(data)[:,30]

# Split the data as training and testing data - 70% train size, 30% test size
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state = None)

#1 Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Accuracy with RF classifier:",accuracy_score(y_test, prediction)) 
fpr,tpr,thresh = roc_curve(y_test,prediction)      
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC

# Plot ROC curve for Random Forest
plt.plot(fpr,tpr,'g',label = 'Random Forest')
plt.legend("Random Forest", loc='lower right')
plt.legend(loc='lower right')
print("Conf matrix RF classifier:",confusion_matrix(y_test,prediction))  #  Generate confusion matrix

#2 Classification using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Accuracy with Log Reg:", accuracy_score(y_test, prediction))
print ("Conf matrix Log Reg:",confusion_matrix(y_test,prediction))
fpr,tpr,thresh = roc_curve(y_test,prediction)
roc_auc = accuracy_score(y_test,prediction)

# Plot ROC curve for Logistic Regression
plt.plot(fpr,tpr,'orange',label = 'Logistic Regression')
plt.legend("Logistic Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')

#3 Classification using SVM
from sklearn.svm import SVC
svc_l = SVC(kernel = "linear", C = 0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
print("Accuracy with SVM-Linear:",accuracy_score(y_test, prediction))
fpr,tpr,thresh = roc_curve(y_test,prediction)
roc_auc = accuracy_score(y_test,prediction)

# Plot ROC curve for SVM-linear
plt.plot(fpr,tpr,'b',label = 'SVM')
plt.legend("SVM", loc ='lower right')
plt.legend(loc ='lower right')
print("Conf matrix SVM-linear:",confusion_matrix(y_test,prediction))

plt.show()

'''
# -------- Apply Recursive Feature Elimination(RFE) and use reduced feature set for prediction ------------------------
# Recursive Feature Elimination(RFE) is a technique that takes entire feature set as input and removes features one at 
# a time up to a specified number or until a stopping criteria is met.
'''
from sklearn.feature_selection import RFE
rfe = RFE(rfc,27)                              
rfe = rfe.fit(X_train, y_train)               # Train RF classifier with only 27 features now
pred = rfe.predict(X_test)

# Test accuracy on reduced data
print("Accuracy by RFClassifier after RFE is applied:", accuracy_score(y_test,pred))

rfe = RFE(svc_l,27)
rfe = rfe.fit(X_train, y_train)               # Train SVM with only 27 features now
pred = rfe.predict(X_test)
print("Accuracy by SVM-Linear after RFE is applied:", accuracy_score(y_test,pred))

rfe = RFE(logreg,27)
rfe = rfe.fit(X_train, y_train)              # Train Logistic-Reg with only 27 features now
pred = rfe.predict(X_test)
print("Accuracy by Logistic Regression after RFE is applied:", accuracy_score(y_test,pred))

############################################ END OF fILE #####################################################################





