#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

from sklearn.externals import joblib

# read data from csv
x_train_raw = pd.read_csv('input/x_train.csv')
y_train_raw = pd.read_csv('input/y_train.csv')
x_test_raw = pd.read_csv('input/x_test.csv')
y_test_raw = pd.read_csv('input/y_test.csv')

x_train = np.array(x_train_raw)
y_train = np.array(y_train_raw)
y_train = y_train.ravel()

x_test = np.array(x_test_raw)
y_test = np.array(y_test_raw)
y_test = y_test.ravel()

# test various classifiers
names = [
        "Nearest Neighbors",
        "Linear SVM", "RBF SVM", "Decision Tree",
        "Random Forest", "AdaBoost", "Gradient Boosting", "Naive Bayes", "Linear Discriminant Analysis",
        "Quadratic Discriminant Analysis",
        "Stochastic Gradient Descent Hinge L2", "Stochastic Gradient Descent Hinge Elasticnet",
        "Stochastic Gradient Descent Modified Huber", "Stochastic Gradient Descent Log"
        ]
classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(),
    SVC(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2"),
    SGDClassifier(loss="hinge", penalty="elasticnet"),
    SGDClassifier(loss="modified_huber", penalty="l2"),
    SGDClassifier(loss="log", penalty="l2")
    ]

# choose the classifier to be keep as a model
classifier_to_export = "Stochastic Gradient Descent Log"

# loop on classifiers
for name, clf in zip(names, classifiers):
    print('//////////////////{0}////////////////////////'.format(name))
    fit_out = clf.fit(x_train, y_train)
    print(fit_out)
    print('    //////////////////SCORE//////////////////')
    score = clf.score(x_test, y_test)
    print(score)
    if name == classifier_to_export:
        print('    //////////////////EXPORT MODEL////////////////////////')
        joblib.dump(clf, 'model/model.pkl')
