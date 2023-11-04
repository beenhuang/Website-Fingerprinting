#!/usr/bin/env python3

"""
<file>    kFP_classifer.py
<brief>   fingerprint using RF, classifiation using kNN
"""

import pickle
import numpy as np
from multiprocessing import Pool, cpu_count

from scipy.spatial.distance import hamming
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

class KFPClassifier():
    def __init__(self, trees=10, m_file=None):
        self.K = 3

        if m_file == None:
            self.fp_generator = RandomForestClassifier(n_estimators=trees, oob_score=True, n_jobs=-1)
            self.knn_finder = NearestNeighbors(n_neighbors=self.K, metric=hamming, n_jobs=-1)
            self.fp_label = None
        else:
            with open(m_file, 'rb') as f:
                self.fp_generator, self.knn_finder, self.fp_label = pickle.load(f)   

    def train(self, X_train, y_train):
        self.fp_generator.fit(X_train, y_train)  
        fp_corpus = self.fp_generator.apply(X_train)
        self.knn_finder.fit(fp_corpus)
        self.fp_label = y_train
    
    def test(self, X_test, label_unmon=0):
        fp_test = self.fp_generator.apply(X_test)
        idxes = self.knn_finder.kneighbors(fp_test, n_neighbors=self.K, return_distance=False)

        y_pred = []
        for idx in idxes:
            kn_labels = [self.fp_label[i] for i in idx]
            pred = kn_labels[0] if len(set(kn_labels)) == 1 else label_unmon
            y_pred.append(pred)

        return y_pred, None
