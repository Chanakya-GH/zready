from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from common import load_split
from metrics import report

Xtr, Xte, ytr, yte = load_split()
# validation slice carved from TRAIN; the test set is never used for any choice
Xfit, Xval, yfit, yval = train_test_split(
    Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)

ratio = (yfit == 0).sum() / (yfit == 1).sum()
configs = [("spw=1", 1.0), ("spw=sqrt", float(np.sqrt(ratio))), ("spw=full", float(ratio))]

best = None
for name, spw in configs:
    clf = XGBClassifier(
        tree_method="hist", n_estimators=1000, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=100, n_jobs=-1, random_state=42,
    )
    clf.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    print(f"{name} ({spw:.1f}): best iteration {clf.best_iteration}, val aucpr {clf.best_score:.4f}")
    report(f"xgb {name}", yte, clf.predict_proba(Xte)[:, 1])
    if best is None or clf.best_score > best[1]:
        best = (clf, clf.best_score, name)

print("selected by validation:", best[2])
out = Path(__file__).parent / "artifacts"
out.mkdir(exist_ok=True)
best[0].save_model(out / "xgb.json")
