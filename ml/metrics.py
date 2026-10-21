import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

def recall_at_precision(y, s, p):
    prec, rec, _ = precision_recall_curve(y, s)
    prec, rec = prec[:-1], rec[:-1]   # drop the final (precision=1, recall=0) sentinel point
    ok = prec >= p
    return float(rec[ok].max()) if ok.any() else 0.0

def report(name, y, s):
    prec, _, _ = precision_recall_curve(y, s)
    levels = " | ".join(f"R@P>={p}: {recall_at_precision(y, s, p):.3f}" for p in (0.5, 0.8, 0.9))
    print(f"[{name}] PR-AUC {average_precision_score(y, s):.4f} | {levels} "
          f"| best precision on any threshold: {prec[:-1].max():.3f}")
    top = np.argsort(-s)[:20]
    print(f"[{name}] frauds among the 20 highest-scored transactions: {int(y[top].sum())}/20")
