import numpy as np
from sklearn import metrics 
from .utils import sigma

def validate(X_test, y_test, W_dict, b_dict, L):
    a = X_test
    for i in range(1, L+1):
        a = sigma(W_dict[i] @ a + b_dict[i], L, i)
    fpr, tpr, _ = metrics.roc_curve(y_test.flatten(), a.flatten(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    a_pred = a.flatten().round().astype(int)
    accuracy = np.mean(a_pred == y_test.flatten())
    return accuracy, auc