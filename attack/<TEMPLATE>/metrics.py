#!/usr/bin/env python3

"""
<file>    metrics.py
<brief>   OW, CW, binary metrics
"""

# open-world score for two classes.`
def ow_score_twoclass(y_true, y_pred, label_unmon):
    tp, fn, tn, fp = 0, 0, 0, 0

    for idx, label_pred in enumerate(y_pred): 
        label_true = y_true[idx]   
        
        if label_true != label_unmon: # positive sample
            if label_pred == label_true: # predict positive
                tp += 1
            else: # predict nagetive
                fn += 1    
        else: # negative sample
            if label_pred == label_true: # predict negative
                tn += 1
            else: # predict positive
                fp += 1
   
    # accuracy
    accuracy = (tp+tn) / float(tp+fn+tn+fp)
    # precision      
    precision = tp / float(tp+fp)
    # recall
    recall = tp / float(tp+fn)
    # F1-score
    f1 = 2*(precision*recall) / float(precision+recall)
    # FPR
    fpr = fp / float(fp+tn)

    # incorrect prediction
    y_true_pred = list(zip(y_true, y_pred))
    pred_incorrect = [x for x in y_true_pred if x[0] != x[1]]
    sorted_incorrect = sorted(pred_incorrect, key=lambda x:x[0]) 

    lines = []
    lines.append(f"[POS] TP: {tp},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"Accuracy: {accuracy}\n")
    lines.append(f"Precision: {precision}\n")
    lines.append(f"Recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"TPR: {recall}\n")
    lines.append(f"FPR: {fpr}\n\n")
    lines.append(f"Incorrect Prediction(y_true, y_pred):{sorted_incorrect}\n\n")
   
    return lines

# open-world score for multi-class.
def ow_score_multiclass(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    # iterate through each element
    for idx, label_pred in enumerate(y_pred): 
        label_true = y_true[idx]

        if label_true != label_unmon: # positive sample
            if label_pred != label_unmon: # predict monitored class
                if label_pred == label_true # predict corrcect website class
                    tp_c += 1
                else: # predict incorrcect website class  
                    tp_i += 1
            else: # predict unmonitored class
                fn += 1    
        else: # negative sample
            if label_pred == label_true: # predict negative
                tn += 1
            else: # predict positive
                fp += 1
 
    # accuracy
    accuracy = (tp_c+tn) / float(tp_c+tp_i+fn+tn+fp)
    # precision      
    precision = tp_c / float(tp_c+tp_i+fp)
    # recall
    recall = tp_c / float(tp_c+tp_i+fn)
    # F1-score
    f1 = 2*(precision*recall) / float(precision+recall)
    # FPR
    fpr = fp / float(fp+tn)

    # incorrect prediction
    y_true_pred = list(zip(y_true, y_pred))
    pred_incorrect = [x for x in y_true_pred if x[0] != x[1]]
    sorted_incorrect = sorted(pred_incorrect, key=lambda x:x[0]) 

    lines = []
    lines.append(f"[POS] TP-c: {tp_c}, TP-i: {tp_i}, FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"Accuracy: {accuracy}\n")
    lines.append(f"Precision: {precision}\n")
    lines.append(f"Recall: {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"TPR: {recall}\n")
    lines.append(f"FPR: {fpr}\n\n")
    lines.append(f"Incorrect Prediction(y_true, y_pred):{sorted_incorrect}\n\n")

    return lines

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

def ow_score_with_th(y_true, y_pred_prob, label_unmon, threshold):
    y_pred = []

    # if the sample's probability is less than the threshold, the sample will get the unmonitord label.
    for x in y_pred_prob:
        label_pred, prob = x

        if prob < threshold: # prob < threshold
            y_pred.append(label_unmon) # get the unmonitord label.
        else:
            y_pred.append(label_pred) # get the prediction label

    return ow_score_multiclass(y_true, y_pred, label_unmon)        

               