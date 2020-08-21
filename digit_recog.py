#0.97992 accuracy on Kaggle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

data_train = pd.read_csv("train.csv")
data_test =  pd.read_csv("test.csv")

y = data_train.iloc[:,0]
X = data_train.iloc[:,1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 30)

#Random Forest -- very quick results
random_classifier =  RandomForestClassifier(random_state=42).fit(X_train,y_train)
RF_score = random_classifier.score(X_test,y_test)
print("\nRandom Forest Accuracy: " + str(RF_score));

#SVM -- slower, but more accurate results
svm = svm.SVC(C= 10,degree = 3,gamma = 'scale',random_state=30).fit(X_train,y_train)
y_pred = svm.predict(X_test)
print("\nSVM Accuracy: " + str(accuracy_score(y_test,y_pred)));

cm = confusion_matrix(y_test,y_pred)
print(cm)

sns.heatmap(cm, annot=True)
plt.show();

#Using the SVM to predict from the test data
X_test_data = data_test.iloc[:,:]
y_test_pred = svm.predict(X_test_data).reshape(-1,1)

sample_data = pd.read_csv(r"C:\Users\isabe\OneDrive\Desktop\digit_recognizer\sample_submission.csv")
preds = pd.DataFrame(y_test_pred,columns=['Label'])
sample_data = sample_data.drop(columns=['Label'])
submit_data = pd.concat([sample_data, preds],axis=1)

submit_data.to_csv('submission.csv',index = False)
