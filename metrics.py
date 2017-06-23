# A small file with various performance metrics written in Python
# Copyright Varun Shenoy, 6/23/17

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
# good for multilabel classification problems
def deepAccuracy(actual, prediction):
    correct = 0.
    total = 0
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == prediction[index][i]):
                correct += 1
            total += 1
    return correct/total
