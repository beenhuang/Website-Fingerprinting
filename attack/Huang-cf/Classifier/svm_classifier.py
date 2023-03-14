#!/usr/bin/env python3

"""
<file>    svm_classifier.py
<brief>   svm classifier
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# training
def train_cf(X_train, y_train, ker="rbf", c=2048, g=0.015625):
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel=ker, C=c, gamma=g))])
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred
