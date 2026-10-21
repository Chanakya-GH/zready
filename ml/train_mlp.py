from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from common import load_split
from metrics import report

torch.manual_seed(42)
np.random.seed(42)

Xtr, Xte, ytr, yte = load_split()
Xfit, Xval, yfit, yval = train_test_split(
    Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)

mean = Xfit.mean(0).astype(np.float32)
std = (Xfit.std(0) + 1e-6).astype(np.float32)

from mlp_model import MLP

def scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()

model = MLP(mean, std)
ratio = float((yfit == 0).sum() / (yfit == 1).sum())
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio ** 0.5))   # moderate weighting, untuned
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

Xf, yf = torch.from_numpy(Xfit), torch.from_numpy(yfit).float()
best, best_state, best_ep, bad = -1.0, None, 0, 0
for ep in range(1, 61):
    model.train()
    perm = torch.randperm(len(Xf))
    for i in range(0, len(perm), 1024):
        idx = perm[i:i + 1024]
        opt.zero_grad()
        loss = lossf(model(Xf[idx]), yf[idx])
        loss.backward()
        opt.step()
    val = average_precision_score(yval, scores(model, Xval))
    if ep % 5 == 0:
        print(f"epoch {ep}: val PR-AUC {val:.4f}")
    if val > best:
        best, best_ep, bad = val, ep, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 15:
            break

model.load_state_dict(best_state)
print(f"best epoch {best_ep}, val PR-AUC {best:.4f}")
report("mlp", yte, scores(model, Xte))

out = Path(__file__).parent / "artifacts"
out.mkdir(exist_ok=True)
torch.save({"state": model.state_dict(), "mean": mean, "std": std}, out / "mlp.pt")
