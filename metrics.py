"""metrics.py - one metrics implementation shared by train.py and evaluate.py."""
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_pred):
    """Class 1 = phishing. FPR = benign URLs wrongly flagged / all benign URLs."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    def ratio(a, b):
        return a / b if b else float("nan")

    return {
        "accuracy": ratio(tp + tn, total),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def print_metrics(m, title):
    print(f"\n== {title} ==")
    print(f"  accuracy             {m['accuracy']:.4f}")
    print(f"  precision            {m['precision']:.4f}")
    print(f"  recall               {m['recall']:.4f}")
    print(f"  false-positive rate  {m['false_positive_rate']:.4f}")
    print(f"  confusion matrix     TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
