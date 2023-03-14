#!/usr/bin/env python3

"""
<file>    ada_classifier.py
<brief>   AdaBoost classifier
"""

from sklearn.ensemble import AdaBoostClassifier


# training
def train_cf(X_train, y_train, trees=100):
    model = AdaBoostClassifier(n_estimators=trees)
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred
