#!/usr/bin/env python3

"""
<file>    svm_classifier.py
<brief>   svm classifier used by CUMUL
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

# training
def train_cumul(X_train, y_train, ker="rbf", c=2048, g=0.015625):
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel=ker, C=c, gamma=g))])
    model.fit(X_train, y_train)

    return model

# test
def test_cumul(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred


def closeworld_score(y_true, y_pred):
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # precision      
    precision = precision_score(y_true, y_pred, average="macro")
    # recall
    recall = recall_score(y_true, y_pred, average="macro")
    # F-score
    f1 = 2*(precision*recall) / float(precision+recall)

    lines = []
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n")

    return lines


# get open-world score
def openworld_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    #print(f"label_unmon: {label_unmon}")

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


## the follow functions are used for finding optimal parameters ##

# in order to find the optimal hyperparameters
# c: 2**11 ~ 2**17
# gamma: 2**-3 ~ 2**3
def find_optimal_hyperparams(X, y):
    #
    model = Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    #hyperparams = {"svc__C":[2048, 4096, 8192, 16384, 32768, 65536, 131072], "svc__gamma":[0.125, 0.25, 0.5, 0, 2, 4, 8]}
    hyperparams = {"svc__C":[256, 526, 1024, 2048], "svc__gamma":[0.0078125, 0.015625, 0.03125, 0.0625]}
    
    clf = GridSearchCV(model, hyperparams, n_jobs=-1, cv=10, scoring=make_scorer(openworld_recall_score, label_unmon=max(y)))
    clf.fit(X, y)
    
    # write to txt file
    lines = []
    lines.append(f"best_params_: {clf.best_params_} \n")
    lines.append(f"best_score_: {clf.best_score_} ")

    return lines

# recall score used by find_optimal_hyperparams() and get_cross_val_score()
def openworld_recall_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FP
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0
    
    #logger.debug(f"label_unmon: {label_unmon}")

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


    # recall
    recall = tp_c / float(tp_c+tp_i+fn)

    return recall



