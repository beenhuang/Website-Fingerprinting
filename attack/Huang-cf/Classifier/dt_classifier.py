#!/usr/bin/env python3

"""
<file>    dt_classifier.py
<brief>   Decision Tree classifier
"""

from sklearn.tree import DecisionTreeClassifier

# training
def train_cf(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred
