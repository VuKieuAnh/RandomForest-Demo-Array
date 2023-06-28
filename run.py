from sklearn.datasets import load_digits
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import numpy as np
from randomforest import RandomForestClassifier
import pandas as pd


digits = load_digits(n_class = 2)
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
# # print(X_train)
# print(y_train)
#
# print("________")
# df = pd.read_csv('data/Breast2classes.csv', header=0)
# df.columns.values[0] = "class"
# y = df[['class']]
# X = df.iloc[:,df.columns !='class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.7)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# forest = RandomForestClassifier()
# forest.fit(X_train, y_train)
#
# accuracy = forest.score(X_test, y_test)
# print ('The accuracy was', 100*accuracy, '% on the test data.')
#
# classifications = forest.predict(X_test)
# print ('The digit at index 0 of X_test was classified as a', classifications[0], '.')
