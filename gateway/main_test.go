package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

func testServer(t *testing.T, infer http.HandlerFunc, timeout time.Duration) *server {
	t.Helper()
	is := httptest.NewServer(infer)
	t.Cleanup(is.Close)
	return &server{
		rdb: redis.NewClient(&redis.Options{Addr: "127.0.0.1:1",
			DialTimeout: 50 * time.Millisecond, MaxRetries: -1}),
		hc: &http.Client{}, inferURL: is.URL, model: "xgb",
		ttl: time.Minute, inferTimeout: timeout,
		logCh: make(chan logEntry, 16),
	}
}

func body(n, f int) *strings.Reader {
	row := "[" + strings.TrimSuffix(strings.Repeat("0,", f), ",") + "]"
	all := "[" + strings.TrimSuffix(strings.Repeat(row+",", n), ",") + "]"
	return strings.NewReader(`{"instances":` + all + `}`)
}

func post(s *server, b *strings.Reader) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	s.handleScore(rec, httptest.NewRequest(http.MethodPost, "/v1/score", b))
	return rec
}

func TestScoreWorksWhenRedisIsDown(t *testing.T) {
	s := testServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"xgb","scores":[0.1,0.9]}`))
	}, time.Second)
	rec := post(s, body(2, nFeatures))
	if rec.Code != 200 || rec.Header().Get("X-Cache") != "MISS" {
		t.Fatalf("got %d, X-Cache=%q", rec.Code, rec.Header().Get("X-Cache"))
	}
	var r scoreResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil || len(r.Scores) != 2 {
		t.Fatalf("bad body: %s", rec.Body.String())
	}
	e := <-s.logCh
	if e.status != 200 || e.n != 2 || e.maxScore == nil || *e.maxScore != 0.9 || e.hit {
		t.Fatalf("unexpected log entry: %+v", e)
	}
}

func TestRejectsWrongFeatureCountWithoutCallingInference(t *testing.T) {
	var calls int32
	s := testServer(t, func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
	}, time.Second)
	if rec := post(s, body(1, 28)); rec.Code != 422 {
		t.Fatalf("want 422, got %d", rec.Code)
	}
	if atomic.LoadInt32(&calls) != 0 {
		t.Fatal("inference was called for an invalid request")
	}
}

func TestRejectsUnknownField(t *testing.T) {
	s := testServer(t, func(http.ResponseWriter, *http.Request) {}, time.Second)
	if rec := post(s, strings.NewReader(`{"instances":[],"extra":1}`)); rec.Code != 400 {
		t.Fatalf("want 400, got %d", rec.Code)
	}
}

func TestInferenceTimeoutIs504(t *testing.T) {
	s := testServer(t, func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(300 * time.Millisecond)
	}, 50*time.Millisecond)
	if rec := post(s, body(1, nFeatures)); rec.Code != 504 {
		t.Fatalf("want 504, got %d", rec.Code)
	}
}

func TestInferenceErrorIs502(t *testing.T) {
	s := testServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(500)
	}, time.Second)
	if rec := post(s, body(1, nFeatures)); rec.Code != 502 {
		t.Fatalf("want 502, got %d", rec.Code)
	}
}

func TestReadyzOnlyDependsOnInference(t *testing.T) {
	s := testServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
	}, time.Second)
	pool, err := pgxpool.New(context.Background(), "postgres://u:p@127.0.0.1:1/db")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	s.pool = pool

	rec := httptest.NewRecorder()
	start := time.Now()
	s.handleReady(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != 200 {
		t.Fatalf("want 200 with redis and postgres down, got %d", rec.Code)
	}
	if d := time.Since(start); d > 700*time.Millisecond {
		t.Fatalf("readyz took %v", d)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["inference"] != "up" || body["redis"] != "down" || body["postgres"] != "down" {
		t.Fatalf("unexpected body: %v", body)
	}
}
