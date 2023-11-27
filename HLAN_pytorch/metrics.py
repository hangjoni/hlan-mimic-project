import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F

def calculate_accuracy(labels_predicted,labels): # this should be same as the recall value
    # turn the multihot representation to a list of true labels
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
        #print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    overlapping = 0
    label_dict = {x: x for x in label_nozero} # create a dictionary of labels for the true labels
    union = len(label_dict)
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            overlapping = overlapping + 1
        else:
            union = union + 1        
    return overlapping / union

def calculate_precision_recall(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    if (len(labels_predicted)==0): # if nothing predicted, then set the precision as 0.
        precision=0
    else: 
        precision = count / len(labels_predicted)
    recall = count / len(label_nozero)
    #fmeasure = 2*precision*recall/(precision+recall)
    #print(count, len(label_nozero))
    return precision, recall

######################################################################################################################            
# the code below is from https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py under the MIT license
######################################################################################################################            
def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1
    
##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

# calculate AUC metrics
def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0: # if the label has a true instance in the data
            fpr[i], tpr[i], _ = metrics.roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = metrics.auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ymic, yhatmic) 
    roc_auc["auc_micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return roc_auc

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
    #numpy.logical_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'logical_and'>
    #Compute the truth value of x1 AND x2 element-wise.

def get_micro_metrics(all_logits, all_labels_actuals, threshold=0.5):
    all_probs = [F.sigmoid(logits) for logits in all_logits]

    preds = np.concatenate(all_probs)
    actuals = np.concatenate(all_labels_actuals)
    preds = np.where(preds > threshold, 1, 0)
    acc, prec, rec, f1 = all_micro(preds.ravel(), actuals.ravel())
        
    return acc, prec, rec, f1