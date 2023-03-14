#!/usr/bin/env python3

"""
<file>    xg_classifier.py
<brief>   xgboost classifier
"""

from xgboost import XGBClassifier

# training
def train_cf(X_train, y_train, trees=100):
    model = XGBClassifier(n_estimators=trees)
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred
