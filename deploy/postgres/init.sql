CREATE TABLE IF NOT EXISTS score_log (
  id           BIGSERIAL PRIMARY KEY,
  ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_hash TEXT NOT NULL,
  model        TEXT NOT NULL,
  n_instances  INT NOT NULL,
  max_score    DOUBLE PRECISION,
  cache_hit    BOOLEAN NOT NULL,
  latency_ms   DOUBLE PRECISION NOT NULL,
  status       INT NOT NULL
);
CREATE INDEX IF NOT EXISTS score_log_ts ON score_log (ts DESC);
