import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC 


dataset = pd.read_csv('breast-cancer.csv')
dataset.shape

"""There are 569 unique instances. There are 31 features on total. The Features are radius mean, texture mean, radius verse."""

dataset.head()

"""Meaning of 'M' is malignant."""

labelencoder = LabelEncoder()
dataset['diagnosis'] = labelencoder.fit_transform(dataset['diagnosis'].values)

train, test = train_test_split(dataset, test_size=0.3)

X_train = train.drop('diagnosis', axis=1)
Y_train = train.loc[:,'diagnosis']

X_test = test.drop('diagnosis', axis=1)
Y_test = test.loc[:, 'diagnosis']

"""With Logistic Regression"""

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

confusion_matrix(Y_test, predictions)
"""113 TrueNegative(TN), 1 FalsePozitive(FP), 1 FalseNegative(FN), 56 TruePozitive(TP) here. That means, 169 out of 171 predictions are correct."""

print("Logistic Regression: ", classification_report(Y_test, predictions))


"""With Support Vector Machine(SVM)"""

model_2 = LinearSVC()
model_2.fit(X_train, Y_train)
predictions = model_2.predict(X_test)

confusion_matrix(Y_test, predictions)
"""113 TrueNegative(TN), 1 FalsePozitive(FP), 1 FalseNegative(FN), 56 TruePozitive(TP) here. That means, 169 out of 171 predictions are correct."""

print("Support Vector Machine: ", classification_report(Y_test, predictions))


"""In this problem, we got the same predictions in Logistic Regression and in Support Vector Machine(SVM)"""