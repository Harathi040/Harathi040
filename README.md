 
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('GGBS.csv')

df.head()

df.shape()

X = df.iloc[:,[2,3,4]]

Y = df.iloc[:, 5]

X.head()

Y.head()

from sklearn.model_selection import train_test_split

X_train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print("Training data : ",X_Train.shape)

print("Training data : ",X_Test.shape) 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_Train = sc_X.fit_transform(X_Train)

X_Test = sc_X.Transform(X_Test)

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_Train, Y_Train)

Y_Pred = Classifier.predict(X_Test)

Y_Pred

from sklearn import metrics

print('Accuracy Score : with linear kernel')

print(metrics.accuracy_score(Y_Test, Y_Pred))

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf')

classifier.fit(X_Train, Y_Train)

Y_Pred = Classifier.predict(X_Test)

print('Accuracy Score : with default rbf kernel')

print(metrics.accuracy_score(Y_Test, Y_Pred))

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', gamma = 15, C=7, random_state=0)

classifier.fit(X_Train, Y_Train)

Y_Pred = Classifier.predict(X_Test)

print('Accuracy Score on Test Data: with default rbf kernel')

print(metrics.accuracy_score(Y_Test, Y_Pred))

svc=SVC(kernel='poly', degree = 4)

svc.fit(X_Train, Y_Train)

y_pred=svc.predict(X_Test)

print('Accuracy SCore:with poly kernel and degree ')

print(metrics.accuracy_score(Y_Test,Y_Pred))

import matplotlib.pyplot as plt

plt.scatter(X_Train[:, 0], X_Train[:, 1],C=Y_Train)

plt.xlabel('CA')

plt.ylabel('FA')

plt.title('Test Data')

plt.show()

import matplotlib.pyplot as plt

plt.scatter(X_Test[:, 0], X_Test[:, 1],C=Y_Test)

plt.xlabel('CA')

plt.ylabel('FA')

plt.title('Test Data')

plt.show()

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)

plt.scatter(X_Test[:, 0], X_Test[:, 1],c=Y_Test)

w = classifier.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(-2.5, 2.5)

yy = a * xx - (classifier.intercept_[0]) / w[1]

plt.plot(xx, yy)

plt.axis("off").plt.show();
