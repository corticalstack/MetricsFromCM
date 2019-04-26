import pandas as pd
import numpy as np
from sklearn.metrics import *

cm = confusion_matrix(self.y_test, self.y_pred)
get_tp_from_cm(cm)
get_tn_from_cm(cm)
get_fp_from_cm(cm)
get_fn_from_cm(cm)


# True positives are the diagonal elements
def get_tp_from_cm(self, cm):
    tp = np.diag(cm)
    print('tp', np.sum(np.diag(cm)))
    return np.sum(tp)


def get_tn_from_cm(self, cm):
    tn = []
    for i in range(self.n_classes):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        tn.append(sum(sum(temp)))
    print('tn ', np.sum(tn))
    return np.sum(tn)


# Sum of columns minus diagonal
def get_fp_from_cm(self, cm):
    fp = []
    for i in range(self.n_classes):
        fp.append(sum(cm[:, i]) - cm[i, i])
    print('fp ', np.sum(fp))
    return np.sum(fp)


# Sum of rows minus diagonal
def get_fn_from_cm(self, cm):
    fn = []
    for i in range(self.n_classes):
        fn.append(sum(cm[i, :]) - cm[i, i])
    print('fn', np.sum(fn))
    return np.sum(fn)
