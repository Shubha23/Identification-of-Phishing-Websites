
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,roc_curve,auc, confusion_matrix
from sklearn import preprocessing

# Read data

dataframe = pd.read_csv('Dataset.csv')

X = np.array(dataframe)[:,0:30]
y = np.array(dataframe)[:,30]

# Split the data as training and testing data

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state=0)

#1 Classification using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print(accuracy_score(y_test, prediction))
fpr,tpr,thresh = roc_curve(y_test,prediction)
roc_auc = accuracy_score(y_test,prediction)
plt.plot(fpr,tpr,'g',label = 'Random Forest')
plt.legend("Random Forest", loc='lower right')
plt.legend(loc='lower right')
print (confusion_matrix(y_test,prediction))

#2 Classification using logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print(accuracy_score(y_test, prediction))
print (confusion_matrix(y_test,prediction))
fpr,tpr,thresh = roc_curve(y_test,prediction)
roc_auc = accuracy_score(y_test,prediction)
plt.plot(fpr,tpr,'orange',label = 'Logistic Regression')
plt.legend("Logistic Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')

#3 Classification using SVM

from sklearn.svm import SVC
svc_l = SVC(kernel="linear", C=0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
print(accuracy_score(y_test, prediction))
fpr,tpr,thresh = roc_curve(y_test,prediction)
roc_auc = accuracy_score(y_test,prediction)
plt.plot(fpr,tpr,'b',label = 'SVM')
plt.legend("SVM", loc='lower right')
plt.legend(loc='lower right')
print (confusion_matrix(y_test,prediction))

plt.show()

#---- FEATURE SELECTION - Recursive Feature Elimination ------------------------

from sklearn.feature_selection import RFE
rfe = RFE(svc_l,27)
rfe = rfe.fit(X_train, y_train)
pred = rfe.predict(X_test)
print(accuracy_score(y_test,pred))


rfe = RFE(rfc,27)
rfe = rfe.fit(X_train, y_train)
pred = rfe.predict(X_test)
print(accuracy_score(y_test,pred))

rfe = RFE(logreg,27)
rfe = rfe.fit(X_train, y_train)
pred = rfe.predict(X_test)
print(accuracy_score(y_test,pred))




