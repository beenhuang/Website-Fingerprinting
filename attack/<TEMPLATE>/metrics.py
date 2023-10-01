#!/usr/bin/env python3

"""
<file>    metrics.py
<brief>   OW, CW, binary metrics
"""

# binary score
def binary_score(y_true, y_pred, label_unmon):
    tp, fn, tn, fp = 0, 0, 0, 0

    # iterate through each element
    for i in range(len(y_pred)):
        # [case_1]: pos sample, predict pos.
        if y_true[i] != label_unmon and y_pred[i] == y_true[i]:
            tp += 1
        # [case_2]: pos sample, predict neg.
        elif y_true[i] != label_unmon and y_pred[i] != y_true[i]:
            fn += 1
        # [case_3]: neg sample, predict neg.    
        elif y_true[i] == label_unmon and y_pred[i] == y_true[i]:
            tn += 1
        # [case_4]: neg sample, predict pos.    
        elif y_true[i] == label_unmon and y_pred[i] != y_true[i]:
            fp += 1   
        else:
            sys.exit(f"ERROR prediction:{y_pred[i]}, true_label:{y_true[i]}")        

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

# open-world score
def openworld_score(y_true, y_pred, label_unmon):
    # TP-correct, TP-incorrect, FN  TN, FN
    tp_c, tp_i, fn, tn, fp = 0, 0, 0, 0, 0

    # iterate through each element
    for i in range(len(y_pred)):
        # [case_1]: pos sample, predict pos and correct.
        if y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] == y_true[i]:
            tp_c += 1
        # [case_2]: pos sample, predict pos but incorrect class.
        elif y_true[i] != label_unmon and y_pred[i] != label_unmon and y_pred[i] != y_true[i]:
            tp_i += 1
        # [case_3]: pos sample, predict neg.
        elif y_true[i] != label_unmon and y_pred[i] == label_unmon:
            fn += 1
        # [case_4]: neg sample, predict neg.    
        elif y_true[i] == label_unmon and y_pred[i] == y_true[i]:
            tn += 1
        # [case_5]: neg sample, predict pos    
        elif y_true[i] == label_unmon and y_pred[i] != y_true[i]:
            fp += 1   
        else:
            sys.exit(f"ERROR prediction:{y_pred[i]}, true_label:{y_true[i]}")        

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
def closedworld_score(y_true, y_pred):
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

def ow_score_with_th(y_true, y_pred_th, label_unmon, threshold):
    y_pred = []

    # if the sample's probability is less than the threshold, the sample will get the unmonitord label.
    for x in y_pred_th:
        if x[1] < threshold:
            y_pred.append(label_unmon)
        else:
            y_pred.append(x[0]) 

    return openworld_score(y_true, y_pred, label_unmon)        

               