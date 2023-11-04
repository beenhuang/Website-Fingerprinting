#!/usr/bin/env python3

"""
<file>    metrics.py
<brief>   open-world, closed-world metrics
"""

from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, RocCurveDisplay, recall_score, PrecisionRecallDisplay

def pr_curve(label, score, label_unmon=0):
    print(f"[PR Curve]mon:{label.count(1)}, unmon:{label.count(0)}") 
    PrecisionRecallDisplay.from_predictions(label, score, drop_intermediate=True) 
    plt.grid()
    plt.show()

def roc_curve(label, score, label_unmon=0):
    print(f"[ROC Curve]mon:{label.count(1)}, unmon:{label.count(0)}")    
    RocCurveDisplay.from_predictions(label, score, drop_intermediate=True) 
    plt.grid()
    plt.show()    

# open-world score for two classes.`
def ow_score_twoclass(y_true, y_pred, label_unmon=0, return_str=True):
    tp, fn, tn, fp = 0, 0, 0, 0

    for idx, pred in enumerate(y_pred): 
        label = y_true[idx]   
        
        if label != label_unmon: # [POSITIVE] sample
            if pred != label_unmon: # predict [POSITIVE]
                tp += 1
            else: # predict [NEGATIVE]
                fn += 1    
        else: # [NEGATIVE] sample
            if pred == label: # predict [NEGATIVE]
                tn += 1
            else: # predict [POSITIVE]
                fp += 1
   
    # metircs
    try:
        accuracy = (tp+tn) / float(tp+fn+tn+fp)
    except ZeroDivisionError:
        accuracy = 0.0
    try:    
        precision = tp / float(tp+fp)
    except ZeroDivisionError:
        precision = 0.0
    try:    
        recall = tp / float(tp+fn)
    except ZeroDivisionError:
        recall = 0.0  
    try:    
        f1 = 2*(precision*recall) / float(precision+recall)
    except ZeroDivisionError:
        f1 = 0.0    
    try:    
        fpr = fp / float(fp+tn)
    except ZeroDivisionError:
        fpr = 0.0    

    if return_str == True:
        return [f"[POS] TP:{tp}, FN:{fn},  POS_total:{tp+fn}\n",
                f"[NEG] TN:{tn}, FP:{fp},  NEG_total:{tn+fp}\n\n",
                f"Accuracy:{accuracy}\n",
                f"Precision:{precision}\n",
                f"Recall:{recall}\n",
                f"F1:{f1}\n",
                f"TPR:{recall}\n",
                f"FPR:{fpr}\n\n"]
    else:
        return recall

# open-world score for multi-class.
def ow_score_multiclass(y_true, y_pred, label_unmon=0, return_str=True):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    for idx, pred in enumerate(y_pred): 
        label = y_true[idx]

        if label != label_unmon: # [POSITIVE] sample
            if pred != label_unmon: # predict [MONITORED] class
                if pred == label: # predict [CORRECT] website class
                    tp_c += 1
                else: # predict [INCORRECT] website class  
                    tp_i += 1
            else: # predict [UNMONITORED] class
                fn += 1    
        else: # [NEGATIVE] sample
            if pred == label: # predict [NEGATIVE]
                tn += 1
            else: # predict [POSITIVE]
                fp += 1

    # metircs
    try:
        accuracy = (tp_c+tn) / float(tp_c+tp_i+fn+tn+fp)
    except ZeroDivisionError:
        accuracy = 0.0
    try:    
        precision = tp_c / float(tp_c+tp_i+fp)
    except ZeroDivisionError:
        precision = 0.0
    try:    
        recall = tp_c / float(tp_c+tp_i+fn)
    except ZeroDivisionError:
        recall = 0.0  
    try:    
        f1 = 2*(precision*recall) / float(precision+recall)
    except ZeroDivisionError:
        f1 = 0.0    
    try:    
        fpr = fp / float(fp+tn)
    except ZeroDivisionError:
        fpr = 0.0  

    # incorrect prediction
    tpi_inst = [x for x in list(zip(y_true, y_pred)) if x[0]!=label_unmon and x[1]!=label_unmon and x[0]!=x[1]]
    tpi_inst_sorted = sorted(tpi_inst, key=lambda x:x[0]) 
    fn_inst = [x for x in list(zip(y_true, y_pred)) if x[0]!=label_unmon and x[1]==label_unmon]
    fn_label = [x[0] for x in fn_inst]
    fp_inst = [x for x in list(zip(y_true, y_pred)) if x[0]==label_unmon and x[1]!=label_unmon]
    fp_pred = [x[1] for x in fp_inst]

    if return_str == True:
        return [f"[POS] TP-c:{tp_c}, TP-i:{tp_i}, FN:{fn}, pos_total:{tp_c+tp_i+fn}\n",
                f"[NEG] TN:{tn}, FP:{fp}, neg_total:{tn+fp}\n\n",
                f"Accuracy:{accuracy}\n",
                f"Precision:{precision}\n",
                f"Recall:{recall}\n",
                f"F1:{f1}\n",
                f"TPR:{recall}\n",
                f"FPR:{fpr}\n\n",
                f"TP-i Instances(label, pred):{tpi_inst_sorted}\n",
                f"FN Instances(label, 0):{Counter(fn_label)}\n",
                f"FP Instances(0, pred):{Counter(fp_pred)}\n\n"]    
    else:
        return recall    

# closed-world score
def cw_score(y_true, y_pred):
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # precision      
    precision = precision_score(y_true, y_pred, average="macro")
    # recall
    recall = recall_score(y_true, y_pred, average="macro")
    # F1-score
    f1 = 2*(precision*recall) / float(precision+recall)

    y_true_pred = list(zip(y_true, y_pred))
    pred_incorrect = [x for x in y_true_pred if x[0] != x[1]]
    sorted_incorrect = sorted(pred_incorrect, key=lambda x:x[0]) 

    lines = []
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall: {recall}\n")
    lines.append(f"F1: {f1}\n\n")
    lines.append(f"incorrect prediction:{sorted_incorrect}")

    return lines

               