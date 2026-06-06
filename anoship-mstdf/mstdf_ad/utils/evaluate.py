import numpy as np

from mstdf_ad.utils.getbestF1 import get_bestF1

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# from utils.spot import SPOT
from sklearn.metrics import roc_auc_score


def evaluate_spot(train_scores, test_scores, labels, q=1e-4):
    spot = SPOT(q=q)

    spot.fit(train_scores, test_scores)
    spot.initialize(level=0.9999)

    results = spot.run(with_alarm=True)

    preds = np.zeros_like(test_scores, dtype=int)
    preds[results['alarms']] = 1

    gt = labels.astype(int)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, preds, average='binary')

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f_score,
        'threshold': spot.extreme_quantile,
        'Alarms': results['alarms']
    }


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


from mstdf_ad.utils.affiliation.generics import convert_vector_to_events
from mstdf_ad.utils.affiliation.metrics import pr_from_events


def compute_affiliation_metrics(
        gt: np.ndarray,
        anomaly_score: np.ndarray,
        threshold: float,
):
    """
    Compute Affiliation Precision / Recall / F1 (KDD 2022).

    Parameters
    ----------
    gt : np.ndarray, shape (T,)
        Ground-truth binary labels (0 or 1).
    anomaly_score : np.ndarray, shape (T,)
        Anomaly scores (higher = more anomalous).
    threshold : float
        Threshold to binarize anomaly scores.

    Returns
    -------
    dict
        {
            'Affiliation_P': float,
            'Affiliation_R': float,
            'Affiliation_F1': float
        }
    """

    # 1. binarize prediction
    pred = (anomaly_score >= threshold).astype(int)

    # 2. vector -> events
    events_pred = convert_vector_to_events(pred.tolist())
    events_gt = convert_vector_to_events(gt.tolist())

    # 3. time range
    Trange = (0, len(gt))

    # 4. affiliation precision & recall
    pr = pr_from_events(events_pred, events_gt, Trange)

    P = pr['precision']
    R = pr['recall']
    F1 = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)

    return {
        'Affiliation_P': float(P),
        'Affiliation_R': float(R),
        'Affiliation_F1': float(F1),
    }


def best_affiliation_f1(lab, scores, n_thresh=200, eps=1e-8):
    """
    Fast best Affiliation-F1 threshold search.
    Philosophy aligned with classic best-F1 selection.
    """

    lab = lab.numpy() if hasattr(lab, "numpy") else np.asarray(lab)
    scores = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)

    T = (0, len(lab))
    gt_events = convert_vector_to_events(lab.tolist())

    # 🔑 use quantile-based thresholds (key difference)
    qs = np.linspace(0.001, 0.999, n_thresh)
    thresholds = np.quantile(scores, qs)

    best = {
        "f1_score": -1,
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0
    }

    for th in thresholds:
        pred = (scores > th).astype(int)
        pred_events = convert_vector_to_events(pred.tolist())
        if not pred_events:
            continue

        pr = pr_from_events(pred_events, gt_events, T)
        P, R = pr["precision"], pr["recall"]
        F1 = 2 * P * R / (P + R + eps)

        if F1 > best["f1_score"]:
            best.update({
                "f1_score": F1,
                "threshold": th,
                "precision": P,
                "recall": R
            })

    return best


def evaluate(init_score, test_score, test_label, anomaly_ratio=1, Affiliation=False, Best=True, PA=True):
    res = {
        'init_score': init_score,
        'test_score': test_score,
        'test_label': test_label,
        'anomaly_ratio': anomaly_ratio,
    }

    auc = roc_auc_score(test_label, test_score)

    res['auc'] = auc

    # res_ent = evaluate_spot(init_score, test_score, test_label)
    # res['precision'] = res_ent['precision']
    # res['recall'] = res_ent['recall']
    # res['f1_score'] = res_ent['f1_score']
    # res['threshold'] = res_ent['threshold']

    if Best == False:
        # get-threshold
        combined_score = np.concatenate([init_score, test_score], axis=0)
        threshold = np.percentile(combined_score, 100 - anomaly_ratio)

        print(threshold)

        if Affiliation == True:
            res_ent = compute_affiliation_metrics(
                gt=test_label,
                anomaly_score=test_score,
                threshold=threshold
            )

            res['precision'] = res_ent['Affiliation_P']
            res['recall'] = res_ent['Affiliation_R']
            res['f1_score'] = res_ent['Affiliation_F1']
            res['threshold'] = threshold
        else:
            pred = (test_score > threshold).astype(int)
            gt = test_label.astype(int)
            if PA == True:
                gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

            res['precision'] = precision
            res['recall'] = recall
            res['f1_score'] = f_score
            res['test_pred'] = pred
            res['threshold'] = threshold

    else:
        if Affiliation == True:
            res_ent = best_affiliation_f1(
                lab=test_label,
                scores=test_score,
            )

            res['precision'] = res_ent['precision']
            res['recall'] = res_ent['recall']
            res['f1_score'] = res_ent['f1_score']
            res['threshold'] = res_ent['threshold']
        else:
            res_ent = get_bestF1(test_label.copy(), test_score.copy(), PA)

            test_pred = (test_score > res_ent['threshold']).astype(int)

            print(res_ent['threshold'])

            res['precision'] = res_ent['precision']
            res['recall'] = res_ent['recall']
            res['f1_score'] = res_ent['f1_score']
            res['test_pred'] = test_pred
            res['threshold'] = res_ent['threshold']
            res['true_anomaly'] = res_ent['true_anomaly']
            res['pred_anomaly'] = res_ent['pred_anomaly']
            res['anomaly_ratio'] = res_ent['anomaly_ratio']

    return res
