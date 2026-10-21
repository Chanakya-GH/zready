package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

const (
	nFeatures = 29
	maxRows   = 1024
	maxBody   = 1 << 20
)

var (
	reqTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "gateway_requests_total", Help: "Score requests by status and cache result."},
		[]string{"status", "cache"})
	reqDur = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "gateway_request_duration_seconds", Help: "End-to-end /v1/score latency.",
		Buckets: prometheus.DefBuckets}, []string{"cache"})
	logDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "gateway_log_dropped_total", Help: "Log rows dropped because the queue was full."})
)

func init() { prometheus.MustRegister(reqTotal, reqDur, logDropped) }

type scoreRequest struct {
	Instances [][]float64 `json:"instances"`
}

type scoreResponse struct {
	Model  string    `json:"model"`
	Scores []float64 `json:"scores"`
}

type logEntry struct {
	hash      string
	model     string
	n         int
	maxScore  *float64
	hit       bool
	latencyMs float64
	status    int
}

type server struct {
	rdb          *redis.Client
	pool         *pgxpool.Pool
	hc           *http.Client
	inferURL     string
	model        string
	ttl          time.Duration
	inferTimeout time.Duration
	logCh        chan logEntry
}

func validate(req scoreRequest) string {
	if len(req.Instances) == 0 || len(req.Instances) > maxRows {
		return "instances must contain 1.." + strconv.Itoa(maxRows) + " rows"
	}
	for _, row := range req.Instances {
		if len(row) != nFeatures {
			return "each instance needs exactly " + strconv.Itoa(nFeatures) + " features"
		}
	}
	return ""
}

func parseScores(b []byte, n int) ([]float64, bool) {
	var r scoreResponse
	if json.Unmarshal(b, &r) != nil || len(r.Scores) != n {
		return nil, false
	}
	return r.Scores, true
}

func maxOf(xs []float64) *float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return &m
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func (s *server) enqueueLog(e logEntry) {
	select {
	case s.logCh <- e:
	default:
		logDropped.Inc()
	}
}

func (s *server) logWorker(done chan<- struct{}) {
	defer close(done)
	for e := range s.logCh {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := s.pool.Exec(ctx,
			`INSERT INTO score_log (request_hash, model, n_instances, max_score, cache_hit, latency_ms, status)
			 VALUES ($1,$2,$3,$4,$5,$6,$7)`,
			e.hash, e.model, e.n, e.maxScore, e.hit, e.latencyMs, e.status)
		cancel()
		if err != nil {
			slog.Warn("log insert failed", "err", err)
		}
	}
}

func (s *server) handleScore(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	status, cache := http.StatusOK, "miss"
	var (
		hash     string
		n        int
		maxScore *float64
	)
	defer func() {
		d := time.Since(start)
		reqTotal.WithLabelValues(strconv.Itoa(status), cache).Inc()
		reqDur.WithLabelValues(cache).Observe(d.Seconds())
		if hash != "" {
			s.enqueueLog(logEntry{hash, s.model, n, maxScore, cache == "hit",
				float64(d.Microseconds()) / 1000, status})
		}
	}()

	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	var req scoreRequest
	if err := dec.Decode(&req); err != nil {
		status = http.StatusBadRequest
		writeErr(w, status, "invalid JSON body")
		return
	}
	if msg := validate(req); msg != "" {
		status = http.StatusUnprocessableEntity
		writeErr(w, status, msg)
		return
	}
	n = len(req.Instances)
	canon, _ := json.Marshal(req.Instances)
	sum := sha256.Sum256(append([]byte(s.model+"|"), canon...))
	hash = hex.EncodeToString(sum[:])
	key := "score:" + hash

	// cache lookup; any Redis failure degrades to a miss
	cctx, ccancel := context.WithTimeout(r.Context(), 200*time.Millisecond)
	cached, err := s.rdb.Get(cctx, key).Bytes()
	ccancel()
	if err == nil {
		if sc, ok := parseScores(cached, n); ok {
			cache = "hit"
			maxScore = maxOf(sc)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache", "HIT")
			_, _ = w.Write(cached)
			return
		}
	} else if !errors.Is(err, redis.Nil) {
		slog.Warn("cache get failed", "err", err)
	}

	// call the inference service
	payload, _ := json.Marshal(req)
	ictx, icancel := context.WithTimeout(r.Context(), s.inferTimeout)
	defer icancel()
	ireq, _ := http.NewRequestWithContext(ictx, http.MethodPost, s.inferURL+"/score", bytes.NewReader(payload))
	ireq.Header.Set("Content-Type", "application/json")
	resp, err := s.hc.Do(ireq)
	if err != nil {
		if errors.Is(ictx.Err(), context.DeadlineExceeded) {
			status = http.StatusGatewayTimeout
			writeErr(w, status, "inference timed out")
		} else {
			status = http.StatusBadGateway
			writeErr(w, status, "inference unreachable")
		}
		return
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	sc, ok := parseScores(out, n)
	if err != nil || resp.StatusCode != http.StatusOK || !ok {
		status = http.StatusBadGateway
		writeErr(w, status, "bad response from inference")
		return
	}
	maxScore = maxOf(sc)

	// best-effort cache fill
	sctx, scancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	_ = s.rdb.Set(sctx, key, out, s.ttl).Err()
	scancel()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	_, _ = w.Write(out)
}

func (s *server) pingInference(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.inferURL+"/readyz", nil)
	if err != nil {
		return err
	}
	resp, err := s.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.New("inference not ready")
	}
	return nil
}

// Ready depends only on inference: cache and logging are best-effort.
func (s *server) handleReady(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), time.Second)
	defer cancel()
	state := func(err error) string {
		if err != nil {
			return "down"
		}
		return "up"
	}
	infErr := s.pingInference(ctx)
	body := map[string]string{
		"inference": state(infErr),
		"redis":     state(s.rdb.Ping(ctx).Err()),
		"postgres":  state(s.pool.Ping(ctx)),
	}
	code := http.StatusOK
	if infErr != nil {
		code = http.StatusServiceUnavailable
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func mustDuration(k, d string) time.Duration {
	v, err := time.ParseDuration(env(k, d))
	if err != nil {
		slog.Error("bad duration", "var", k, "err", err)
		os.Exit(1)
	}
	return v
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, env("DATABASE_URL", "postgres://zready:zready@localhost:5432/zready"))
	if err != nil {
		slog.Error("postgres config", "err", err)
		os.Exit(1)
	}
	defer pool.Close()

	s := &server{
		rdb: redis.NewClient(&redis.Options{
			Addr: env("REDIS_ADDR", "localhost:6379"),
			DialTimeout: 500 * time.Millisecond, ReadTimeout: 200 * time.Millisecond,
			WriteTimeout: 200 * time.Millisecond,
		}),
		pool:         pool,
		hc:           &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 32}},
		inferURL:     env("INFERENCE_URL", "http://localhost:8000"),
		model:        env("MODEL_NAME", "xgb"),
		ttl:          mustDuration("CACHE_TTL", "5m"),
		inferTimeout: mustDuration("INFERENCE_TIMEOUT", "2s"),
		logCh:        make(chan logEntry, 1024),
	}
	done := make(chan struct{})
	go s.logWorker(done)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/score", s.handleScore)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.Handle("GET /metrics", promhttp.Handler())

	srv := &http.Server{
		Addr: ":" + env("PORT", "8080"), Handler: mux,
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second,
		WriteTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second,
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server failed", "err", err)
			stop()
		}
	}()
	slog.Info("gateway listening", "addr", srv.Addr, "model", s.model, "inference", s.inferURL)

	<-ctx.Done()
	sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(sctx); err != nil {
		slog.Error("shutdown", "err", err)
	} else {
		close(s.logCh) // safe: no handlers are still running
		<-done
	}
}
