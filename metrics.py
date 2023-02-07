import numpy as np
from fairlearn.reductions import DemographicParity, ErrorRate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

def divide_non_0(a,b):
    res = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return res.item() if res.shape == () else res

def get_metric_function(metric_f):
    def f(X, Y, S, y_pred):
        return metric_f(y_true=Y, y_pred=y_pred>=.5, zero_division=0)
    return f


def getViolation(X, Y, S, predict_method):
    disparity_moment = DemographicParity()
    disparity_moment.load_data(X, Y, sensitive_features=S)
    return disparity_moment.gamma(predict_method).max()


def getError(X, Y, S, predict_method):
    error = ErrorRate()
    error.load_data(X, Y, sensitive_features=S)
    return error.gamma(predict_method)[0]


def di(X, Y, S, y_pred):
    y_pred = y_pred >=.5
    s_values = np.unique(S)
    s_values.sort()
    group_0_mask = S == s_values[0]
    group_1_mask = S == s_values[1]
    PrY1_S0 = np.sum(y_pred[group_0_mask] == 1) / np.sum(group_0_mask)
    PrY1_S1 = np.sum(y_pred[group_1_mask] == 1) / np.sum(group_1_mask)
    disparate_impact = divide_non_0(PrY1_S0, PrY1_S1)
    return disparate_impact


def trueRateBalance(X, Y, S, y_pred):
    y_pred = y_pred >= .5
    s_values = np.unique(S)
    s_values.sort()
    mask_0 = S == s_values[0]
    mask_1 = S == s_values[1]
    results = {}
    for turn_mask, group in zip([mask_1, mask_0], [1, 0]):
        TN, FP, FN, TP = confusion_matrix(Y[turn_mask], y_pred[turn_mask] == 1).ravel()
        results[f'TPR_{group}'] = TP / (TP + FN)
        results[f'TNR_{group}'] = TN / (FP + TN)
    return results

def TPRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return np.abs(rates_dict['TPR_1'] - rates_dict['TPR_0']) # TPRB

def TNRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return np.abs(rates_dict['TNR_1'] - rates_dict['TNR_0'])  # TNRB


default_metrics_dict = {'error': getError,
                'violation': getViolation,
                'di': di,
                'TPRB': TPRB,
                'TNRB': TNRB,
                'f1': get_metric_function(f1_score),
                'precision': get_metric_function(precision_score),
                'recall': get_metric_function(recall_score)
                        }