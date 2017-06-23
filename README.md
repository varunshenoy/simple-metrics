# simple-metrics
A light Python script that can compute accuracy, deep accuracy, precision, recall, and f1 score.

## How to use
All methods take 2 parameters, the actual results and the predicted results for a given task. `deepAccuracy` is mainly for multilabel classification problems, and will fail if the two input arrays are not 2-dimensional. 
