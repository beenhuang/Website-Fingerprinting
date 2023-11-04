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
    def __init__(self, trees=1000, k=3, m_file=None):
        if m_file == None:
            self.fp_generator = RandomForestClassifier(n_estimators=trees, oob_score=True, n_jobs=-1)
            self.knn_finder = NearestNeighbors(n_neighbors=k, metric=hamming, n_jobs=1)
            self.fp_label = None
        else:
            with open(m_file, 'rb') as f:
                self.fp_generator, self.fp_corpus, self.fp_label = pickle.load(f)   

    def train(self, X_train, y_train):
        self.fp_generator.fit(X_train, y_train) # train the fingerprint generator.
        fp_corpus = self.fp_generator.apply(X_train)
        self.knn_finder.fit(fp_corpus) # knn
        self.fp_label = y_train # fingeprints' labels
    
    def test(self, X_test):
        fp_test = self.fp_generator.apply(X_test)
        params = [[x, self.knn_finder, self.fp_label] for x in fp_test]
        
        with Pool(cpu_count()) as pool:
            y_pred = pool.starmap(KFPClassifier.predict, params)

        return y_pred, None
    
    @staticmethod
    def predict(fp, knn_finder, fp_label, label_unmon=0):   
        idxes = knn_finder.kneighbors(np.expand_dims(fp, axis=0), return_distance=False)
        k_closest_labels = [fp_label[i] for i in np.squeeze(idxes, 0)]
        pred = k_closest_labels[0] if len(set(k_closest_labels)) == 1 else label_unmon

        print(f"pred: {pred}", end="\r", flush=True)
        return pred
