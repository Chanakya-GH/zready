import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from common import load_split

ROOT = Path(__file__).resolve().parent.parent
_, Xte, _, yte = load_split()
idx = np.concatenate([np.where(yte == 1)[0][:2], np.where(yte == 0)[0][:3]])
X = Xte[idx]
fx = {"instances": X.tolist(), "labels": yte[idx].tolist(), "expected": {}}
for name, i in {"logreg": 1, "xgb": 1, "mlp": 0}.items():
    sess = ort.InferenceSession(str(ROOT / "models" / f"{name}.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X})[i]
    fx["expected"][name] = (out[:, 1] if out.ndim == 2 else out).tolist()
(ROOT / "inference" / "tests" / "fixture.json").write_text(json.dumps(fx))
print("fixture written:", len(idx), "rows")
