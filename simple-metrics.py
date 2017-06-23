# A small file with various performance metrics written in Python
# Copyright Varun Shenoy, 6/23/17

import numpy as np

# precision is the specificity
# "Specificity relates to the test's ability to correctly detect patients without a condition."
def precision(actual, prediction):
    fp = 0.
    tp = 0.
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == 1 and prediction[index][i] == 1):
                tp += 1
            elif (actual[index][i] == 0 and prediction[index][i] == 1):
                fp += 1
    #print("tp: " + str(tp))
    #print("fp: " + str(fp))
    return tp/(tp + fp)

# recall is the sensitivity
# "Sensitivity refers to the test's ability to correctly detect patients who do have the condition."
def recall(actual, prediction):
    fn = 0.
    tp = 0.
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == 1 and prediction[index][i] == 1):
                tp += 1
            elif (actual[index][i] == 1 and prediction[index][i] == 0):
                fn += 1
    #print("tp: " + str(tp))
    #print("fn: " + str(fn))
    return tp/(tp + fn)

def f1score(actual, prediction):
    prec = precision(actual, prediction)
    rec = recall(actual, prediction)
    f1 = 2 * ((prec * rec)/(prec + rec))
    return f1

# accuracy goes through each array in the actual and prediction arrays.
def accuracy(actual, prediction):
    correct = 0.
    for (index, val) in enumerate(actual):
        if (actual[index] == prediction[index]):
            correct += 1
    return correct/len(actual)

# exact accuracy goes through each integer in each array in the actual and prediction arrays.
# good for multilabel classification problems (this is the same as the Exact Match metric)
def deep_accuracy(actual, prediction):
    correct = 0.
    total = 0
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == prediction[index][i]):
                correct += 1
            total += 1
    return correct/total

# The Hamming score, also called accuracy in the multi-label setting, 
# is defined as the number of correct labels divided by the union of predicted and true labels.
def hamming_score(actual, prediction, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(actual.shape[0]):
        set_true = set( np.where(actual[i])[0] )
        set_pred = set( np.where(prediction[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)
