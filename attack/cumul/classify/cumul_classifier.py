#!/usr/bin/env python3

"""
<file>    svm_classifier.py
<brief>   svm classifier used by CUMUL
"""

import pickle

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

from metrics import ow_score_multiclass

class CumulClassifier():
    def __init__(self, kernel='rbf', c=2048, g=0.015625, m_file=None):
        if m_file == None:
            self.model = Pipeline([("stdscaler", StandardScaler()), ("clf", SVC(kernel=kernel, C=c, gamma=g))])  
        else:
            with open(m_file, 'wb') as f:
                self.model = pickle.load(f)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train) 

    def test(self, X_test):
        y_pred = self.model.predict(X_test) 
        #pos_score = self.model.predict_proba(X_test)
        
        return y_pred, None       

    # c: 2**11, ... , 2**17
    # gamma: 2**-3, ... , 2**3
    @staticmethod
    def find_optimal_hyperparams(X, y):
        model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
        hyperparams = {"svc__C":[2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17], "svc__gamma":[2**-3, 2**-2, 2**-1, 0, 2**1, 2**2, 2**3]}
        #hyperparams = {"svc__C":[256, 526, 1024, 2048], "svc__gamma":[0.0078125, 0.015625, 0.03125, 0.0625]}
        #hyperparams = {"svc__C":[2**11, 2**12], "svc__gamma":[2**-3, 2**-2]}
        
        clf = GridSearchCV(model, hyperparams, n_jobs=-1, cv=10, scoring=make_scorer(ow_score_multiclass, return_str=False))
        clf.fit(X, y)
        
        print(f"best_params_:{clf.best_params_}")
        print(f"best_score_:{clf.best_score_}")

