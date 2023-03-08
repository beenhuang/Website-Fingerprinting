#!/usr/bin/env python3

"""
<file>    kFP_classifer.py
<brief>   fingerprint using RF, classifiation using kNN
"""

import sys
import numpy as np
import multiprocessing as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_kfp(X_train, y_train, trees=1000):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=trees, oob_score=True)
    model.fit(X_train, y_train)

    fingerprints = model.apply(X_train)
    labeled_fps = [[fingerprints[i], y_train[i]] for i in range(len(fingerprints))]

    return model, labeled_fps


def test_kfp(model, labeled_fps, X_test):
    pred_labels = []

    new_fps = model.apply(X_test)

    argus = [[labeled_fps, new_fp] for new_fp in new_fps]

    with mp.Pool(mp.cpu_count()) as pool:
        pred_labels = pool.starmap(knn_classifier, argus)

    return pred_labels

# KNN: k is 6 by default.
def knn_classifier(labeled_fps, new_fp, k=3):
    new_fp = np.array(new_fp, dtype=np.int32)
        
    hamming_dists=[]

    for elem in labeled_fps:
        labeled_fp = np.array(elem[0], dtype=np.int32)
        pred_label = elem[1]

        hamming_distance = np.sum(new_fp != labeled_fp) / float(labeled_fp.size)

        if hamming_distance == 1.0:
                 continue

        hamming_dists.append((hamming_distance, pred_label))


    by_distance = sorted(hamming_dists)
    k_nearest_labels = [p[1] for p in by_distance[:k]]
    majority_vote = max(set(k_nearest_labels), key=k_nearest_labels.count)


    return majority_vote

# get open-world score
def openworld_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    #logger.info(f"label_unmon: {label_unmon}")

    # traverse preditions
    for i in range(len(y_pred)):
        # [case_1]: positive sample, and predict positive and correct.
        if y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] == y_true[i]:
            tp_c += 1
        # [case_2]: positive sample, predict positive but incorrect class.
        elif y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] != y_true[i]:
            tp_i += 1
        # [case_3]: positive sample, predict negative.
        elif y_true[i] != label_unmon and y_pred[i] == label_unmon:
            fn += 1
        # [case_4]: negative sample, predict negative.    
        elif y_true[i] == label_unmon and y_pred[i] == y_true[i]:
            tn += 1
        # [case_5]: negative sample, predict positive    
        elif y_true[i] == label_unmon and y_pred[i] != y_true[i]:
            fp += 1   
        else:
            sys.exit(f"[ERROR]: {y_pred[i]}, {y_true[i]}")        

    # accuracy
    accuracy = (tp_c+tn) / float(tp_c+tp_i+fn+tn+fp)
    # precision      
    precision = tp_c / float(tp_c+tp_i+fp)
    # recall or TPR
    recall = tp_c / float(tp_c+tp_i+fn)
    # F1-score
    f1 = 2*(precision*recall) / float(precision+recall)
    # FPR
    fpr = fp / float(fp+tn)

    lines = []
    lines.append(f"[POS] TP-c: {tp_c},  TP-i(incorrect class): {tp_i},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"TPR: {recall}\n")
    lines.append(f"FPR: {fpr}\n\n\n")

    return lines