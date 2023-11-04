#!/usr/bin/env python3

"""
<file>    wa_classifier.py
<brief>   
"""

import numpy as np
from random import uniform
from multiprocessing import Pool, cpu_count

from sklearn.neighbors import NearestNeighbors

class WaClassifier():
    def __init__(self, feat_len):
        self.k_reco = 5 # Num of neighbors for weight learning
        self.k_neighbors = 5

        self.weights = [uniform(0.5, 1.5) for _ in range(feat_len)]
        self.kNN_finder = NearestNeighbors(n_neighbors=self.k_neighbors, metric=calculate_dist, n_jobs=-1)
        self.label = None

        #print(f'init_weights:{self.weights}')

    def train(self, X_train, y_train, X_target, y_target):
        self.kNN_finder.fit(X_train)
        self.label = y_train

        for tgt_p, tgt_p_label in zip(X_target, y_target):
            print(f"target point's label:{tgt_p_label}", end="\r", flush=True)
            # weight recommendation
            point_bad, dist_bad = self.__point_badness(tgt_p, tgt_p_label, X_train, y_train)
            if point_bad == None:
                print(f"point_bad is None")
                continue
            
            # weight adjustment
            self.__adjust_weights(point_bad, dist_bad)

    def test(self, X_test):
        params = [[x, self.kNN_finder, self.label] for x in X_test]
        
        with Pool(cpu_count()) as pool:
            y_pred = pool.starmap(WaClassifier.predict, params)

        return y_pred, None

    @staticmethod
    def predict(p, kNN_finder, weights, labels, label_unmon=0):   
        idxes = kNN_finder.kneighbors(np.expand_dims(p, axis=0), return_distance=False)
        neighbor_labels = [labels[i] for i in np.squeeze(idxes, 0)]
        print(f"neighbor_labels:{neighbor_labels}")
        pred = kn_labels[0] if len(set(kn_labels)) == 1 else label_unmon
        
        print(f"pred:{pred}", end="\r", flush=True)
        return pred

    def __point_badness(self, tgt_p, tgt_p_label, X_train, y_train):
        idxes = self.kNN_finder.kneighbors(np.expand_dims(tgt_p, axis=0), n_neighbors=len(y_train), return_distance=False)
        closest_data, closest_labels = [X_train[i] for i in np.squeeze(idxes, 0)], [y_train[i] for i in np.squeeze(idxes, 0)] # exclude target point itself.
        reco_good, reco_bad = self.__closest_reco_points(tgt_p_label, closest_data, closest_labels)
        
        if len(reco_good) < self.k_reco or len(reco_bad) < self.k_reco:
            return None, None

        point_bad, dist_bad = [], []
        for i, one_feat in enumerate(tgt_p):
            max_good = max([abs(one_feat-x[i]) for x in reco_good])
            bad_dist = [abs(one_feat-x[i]) for x in reco_bad]
            
            p_bad = len([x for x in bad_dist if x<max_good])
            point_bad.append(p_bad) # number of point badness 

            dist_bad.append(sum(bad_dist)) # sum of distances between target point and reco_bad points

        return point_bad, dist_bad

    def __adjust_weights(self, point_bad, dist_bad):
        min_bad = min(point_bad)

        c1, min_dist_total = 0.0, 0.0
        for i, w in enumerate(self.weights):
            if point_bad[i] != min_bad:
                subtract = w * 0.02 * point_bad[i] / float(self.k_reco)
                self.weights[i] -= subtract
                c1 += subtract * dist_bad[i];
            elif point_bad[i] == min_bad and self.weights[i] > 0:
                min_dist_total += dist_bad[i]

        if min_dist_total != 0.0 :
            for i, w in enumerate(self.weights):
                if point_bad[i] == min_bad and self.weights[i] > 0:         
                    self.weights[i] += c1 / min_dist_total
        
        #print(f'point_bad:{point_bad}, dist_bad:{dist_bad}')
        #print(f'weigts:{self.weights}')        
                    
    def __closest_reco_points(self, p_label, data, labels):
        reco_good, reco_bad = [], []

        for d,l in zip(data,labels):
            if len(reco_good) == self.k_reco and len(reco_bad) == self.k_reco:
                break

            if p_label == l and len(reco_good) < self.k_reco:
                reco_good.append(d)
            elif p_label != l and len(reco_bad) < self.k_reco:
                reco_bad.append(d)

        #print(f'reco_good:{len(reco_good)}, reco_bad:{len(reco_bad)}')        
        return reco_good, reco_bad

    @staticmethod
    def calculate_dist(u, v, weights):
        dist = 0.0
        for i, weight in enumerate(weights):
            if u[i] != -1 and v[i] != -1:
                dist += weight * abs(u[i] - v[i])

        return dist   

if __name__ == '__main__':
    pass
