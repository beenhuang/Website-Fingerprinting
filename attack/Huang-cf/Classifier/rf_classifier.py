#!/usr/bin/env python3

"""
<file>    rf_classifier.py
<brief>   random forest classifier
"""

from sklearn.ensemble import RandomForestClassifier

# training
def train_cf(X_train, y_train, trees=1000, crit="gini"):
    model = RandomForestClassifier(n_estimators=trees, criterion=crit)
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred