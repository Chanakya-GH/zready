from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import load_split
from metrics import report

Xtr, Xte, ytr, yte = load_split()
print(f"train {Xtr.shape}, test {Xte.shape}, fraud rate {ytr.mean():.4%}")

clf = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(Xtr, ytr)
report("logreg", yte, clf.predict_proba(Xte)[:, 1])

from pathlib import Path
import joblib
out = Path(__file__).parent / "artifacts"
out.mkdir(exist_ok=True)
joblib.dump(clf, out / "logreg.joblib")

from pathlib import Path
import joblib
out = Path(__file__).parent / "artifacts"
out.mkdir(exist_ok=True)
joblib.dump(clf, out / "logreg.joblib")
