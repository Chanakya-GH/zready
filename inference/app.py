import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

N_FEATURES = 29
OUT_IDX = {"logreg": 1, "xgb": 1, "mlp": 0}   # which ONNX output holds the fraud score
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"

REQS = Counter("inference_requests_total", "Score requests", ["model", "status"])
LAT = Histogram("inference_latency_seconds", "ONNX Runtime latency", ["model"])


class ScoreRequest(BaseModel):
    instances: list[list[float]] = Field(min_length=1, max_length=1024)

    @field_validator("instances")
    @classmethod
    def row_length(cls, rows):
        if any(len(r) != N_FEATURES for r in rows):
            raise ValueError(f"each instance needs exactly {N_FEATURES} features")
        return rows


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        name = os.getenv("MODEL_NAME", "xgb")
        if name not in OUT_IDX:
            raise RuntimeError(f"unknown MODEL_NAME {name!r}; choose from {sorted(OUT_IDX)}")
        path = Path(os.getenv("MODEL_DIR", DEFAULT_DIR)) / f"{name}.onnx"
        app.state.name = name
        app.state.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        yield

    app = FastAPI(title="zready-inference", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        return {"status": "ready", "model": request.app.state.name}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/score")
    def score(req: ScoreRequest, request: Request):
        st = request.app.state
        X = np.asarray(req.instances, dtype=np.float32)
        if not np.isfinite(X).all():
            REQS.labels(st.name, "422").inc()
            raise HTTPException(422, "non-finite feature values")
        t0 = time.perf_counter()
        out = st.sess.run(None, {"input": X})[OUT_IDX[st.name]]
        LAT.labels(st.name).observe(time.perf_counter() - t0)
        REQS.labels(st.name, "200").inc()
        scores = out[:, 1] if out.ndim == 2 else out
        return {"model": st.name, "scores": scores.tolist()}

    return app


app = create_app()
