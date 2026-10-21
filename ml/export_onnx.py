from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import average_precision_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as XFloat
from xgboost import XGBClassifier
from common import load_split
from mlp_model import MLP

ART = Path(__file__).parent / "artifacts"
_, Xte, _, yte = load_split()

def onnx_scores(path, X, out_idx):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X})[out_idx]
    return out[:, 1] if out.ndim == 2 else out

def check(name, native, onnx_s, tol=1e-4):
    diff = float(np.max(np.abs(native - onnx_s)))
    print(f"[{name}] max |native - onnx| = {diff:.2e} | PR-AUC native "
          f"{average_precision_score(yte, native):.4f} vs onnx "
          f"{average_precision_score(yte, onnx_s):.4f}")
    assert diff < tol, f"{name}: parity check failed ({diff:.2e} >= {tol})"

# 1) logistic regression pipeline
lr = joblib.load(ART / "logreg.joblib")
onx = convert_sklearn(lr, initial_types=[("input", FloatTensorType([None, 29]))],
                      options={id(lr.steps[-1][1]): {"zipmap": False}})
(ART / "logreg.onnx").write_bytes(onx.SerializeToString())
check("logreg", lr.predict_proba(Xte)[:, 1], onnx_scores(ART / "logreg.onnx", Xte, 1))

# 2) XGBoost
xgb = XGBClassifier()
xgb.load_model(ART / "xgb.json")
onx = convert_xgboost(xgb, initial_types=[("input", XFloat([None, 29]))])
(ART / "xgb.onnx").write_bytes(onx.SerializeToString())
check("xgb", xgb.predict_proba(Xte)[:, 1], onnx_scores(ART / "xgb.onnx", Xte, 1))

# 3) PyTorch MLP (sigmoid folded in, standardisation baked in)
ckpt = torch.load(ART / "mlp.pt", weights_only=False)
mlp = MLP(ckpt["mean"], ckpt["std"])
mlp.load_state_dict(ckpt["state"])
mlp.eval()

class Scored(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return torch.sigmoid(self.m(x))

with torch.no_grad():
    native = Scored(mlp)(torch.from_numpy(Xte)).numpy()
    torch.onnx.export(Scored(mlp), torch.from_numpy(Xte[:2]), str(ART / "mlp.onnx"),
                      input_names=["input"], output_names=["score"],
                      dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
                      opset_version=17, dynamo=False)
check("mlp", native, onnx_scores(ART / "mlp.onnx", Xte, 0))
print("all parity checks passed")
