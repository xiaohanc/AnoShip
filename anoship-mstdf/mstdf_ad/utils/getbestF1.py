import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import iqr


def get_bestF1(lab, scores, PA=False):
    scores = scores.numpy() if torch.is_tensor(scores) else scores
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones

    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    new_scores = scores[sortid]

    if PA:
        lab_diff = np.insert(lab, len(lab), 0) - np.insert(lab, 0, 0)
        a_st = np.arange(len(lab) + 1)[lab_diff == 1]
        a_ed = np.arange(len(lab) + 1)[lab_diff == -1]

        thres_a = np.array([np.max(scores[a_st[i]:a_ed[i]]) for i in range(len(a_st))])
        sort_a_id = np.flip(np.argsort(thres_a))
        cum_a = np.cumsum(a_ed[sort_a_id] - a_st[sort_a_id])

        last_thres = np.inf
        TPs = np.zeros_like(new_lab)
        for i, a_id in enumerate(sort_a_id):
            TPs[(thres_a[a_id] <= new_scores) & (new_scores < last_thres)] = cum_a[i - 1] if i > 0 else 0
            last_thres = thres_a[a_id]
        TPs[new_scores < last_thres] = cum_a[-1]
    else:
        TPs = np.cumsum(-new_lab) + ones

    FPs = np.cumsum(new_lab - 1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs

    N = len(lab) - np.flip(TPs > 0).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)

    if PA:
        FPRs = np.insert(FPRs, -1, 0)
        TPRs = np.insert(TPRs, -1, 0)
        AUC = ((TPRs[:-1] + TPRs[1:]) * (FPRs[:-1] - FPRs[1:])).sum() * 0.5
    else:
        AUC = roc_auc_score(lab, scores)

    anomaly_ratio = ones / len(lab)

    return {'AUC': AUC, 'f1_score': F1s[maxid], 'threshold': new_scores[maxid], 'recall': TPRs[maxid], 'precision': PPVs[maxid],
            'maxid': maxid, 'true_anomaly': ones, 'pred_anomaly': TPs[maxid], 'anomaly_ratio': anomaly_ratio}