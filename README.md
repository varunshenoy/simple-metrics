# simple-metrics
A light Python script that can compute accuracy, deep accuracy, precision, recall, and f1 score.

## How to use
Be sure to `import simple-metrics` at the top of your personal file and `simple-metrics.py` is in the same directory. <br><br>
All methods take 2 parameters, the actual results and the predicted results for a given task. `deep_accuracy (aka Exact Match)` and `hamming_score` is mainly for multilabel classification problems, and will fail if the two input arrays are not 2-dimensional. 
