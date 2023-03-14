#!/usr/bin/env python3

"""
<file>    knn_classifier.py
<brief>   k-NN classifier
"""

from sklearn.neighbors import KNeighborsClassifier

# training
def train_cf(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model

# test
def test_cf(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred
