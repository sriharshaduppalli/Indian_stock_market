# Indian_stock_market

Indian stock market data, analysis, prompts, queries, and LLM model scaffold with production-hardening controls.

## What is implemented
- High-level architecture in `ARCHITECTURE.md`
- Initial local assistant scaffold for Indian stock market Q&A:
  - Intent classification
    - Covers general Indian stock market, NSE/BSE/SEBI context, stock analysis, market calculations, and prediction guidance intents
  - Domain knowledge retrieval
    - Hybrid retrieval (keyword + semantic overlap) with metadata filtering
    - ISIN/company-level symbol/entity resolution via enterprise instrument master
    - `SentenceTransformerEmbeddingProvider` for real semantic embeddings (optional; falls back to hash embeddings)
    - `MLReranker` — scikit-learn logistic regression reranker trained self-supervised on knowledge-base items; weights stored as JSON; pure-Python inference (scikit-learn only needed at training time)
    - `KnowledgeBase.refresh_index()` to rebuild embeddings in-place
  - Grounded response composition with source citations
  - Production-style response metadata (confidence, citations, disclaimer, safety flag)
  - Prediction-intent path with multi-horizon signals (`prediction_signals` in response):
    - `PredictionEngine` scores context items for bullish/bearish signals
    - Intraday, swing (1–5 days), and medium-term (1–3 months) directions, probabilities, and rationale
  - Deterministic calculation support for common return/CAGR prompts plus pandas-ta indicator routing in market calculations
  - SEBI-aligned safety checks, prompt-injection defenses, and policy audit logs
  - Enterprise data layer scaffold with source hierarchy, validation, refresh, and lineage metadata
  - Android-ready chat serving scaffold (stable contract, cache, rate-limit, retries, circuit-breaker, fallback)
  - Evaluation release-gate scaffold for factuality, groundedness, hallucination, routing, and safety metrics
  - `BenchmarkSuite` with 9 domain seed cases; `cli benchmark` subcommand for CI-friendly reporting
  - Versioned release registry + rollback target helper
  - Daily continual-learning feedback hook with `DailyFeedbackAnalyzer` (TSV + JSON-line log support)
  - Nightly index refresh daemon (`nightly_refresh_enabled`/`nightly_refresh_hour_utc`) + `trigger_index_refresh()`
  - `POST /admin/refresh` HTTP endpoint for ops-triggered knowledge-base rebuilds (admin token gated)
  - LoRA fine-tuning infrastructure: `QAPairBuilder` (knowledge base → instruction-tuning records) + `LoRAFineTuner` (HuggingFace transformers + peft; optional)
  - `cli train` subcommand for building QA pairs and running LoRA fine-tuning
  - Fast latency mode configuration
  - Frozen integration contract (`v1`) via `ChatApi`
  - Tenant-aware auth + rate limiting in serving layer
  - Data readiness release blockers for stale/partial/incomplete source states
  - Live connector mode for NSE/BSE/regulatory/news datasets via runtime endpoints
  - Pluggable monitoring backend integration (none/logging/http)
  - Environment-driven runtime/deployment config overrides
  - Automated canary + regression + rollout promotion flow
  - Pluggable embedding/reranker/model endpoints with fallback orchestration
  - Async monitoring + feedback logging path and optional background data refresh
  - Distributed-ready cache/rate-limit state backend and tenant key rotation support
  - SRE readiness primitives (p95/p99 latency, error budget signals, runbook mapping)

  ## Production scope and SLOs
  - Allowed use cases: grounded Q&A and risk-aware guidance
  - Disallowed use cases: trade execution and guaranteed-return advice
  - API contract: `v1` response schema for downstream chat-box integrations
  - SLO defaults:
    - max latency: 1200ms
    - min uptime: 99.5%
    - min groundedness: 0.85
    - safety compliance: 0.98
    - max failure rate: 0.1

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Optional open-source NLP and market-data integrations:
   ```bash
   pip install -r requirements-optional.txt
   ```
2. Run tests:
   ```bash
   PYTHONPATH=src pytest -q
   ```
3. Ask a query:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli "What are valuation risks in Indian IT stocks?"
   ```
4. API-friendly JSON output:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli --json "Predict NIFTY next week"
   ```
5. Run the benchmark suite:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli benchmark
   PYTHONPATH=src python -m indian_stock_llm.cli benchmark --json
   ```
6. Build QA pairs for fine-tuning:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli train --output qa_pairs.json
   # With LoRA fine-tuning (requires transformers + peft):
   PYTHONPATH=src python -m indian_stock_llm.cli train --output qa_pairs.json \
       --finetune --base-model google/flan-t5-base --lora-output ./adapter
   ```
7. Run HTTP wrapper for Android/backend integrations:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.http_server
   ```
   - `GET /health` (liveness + contract version)
   - `GET /metrics` (internal/admin endpoint; optional `X-Admin-Token`)
   - `POST /query` (tenant-scoped query with `X-Tenant-Id`, `X-API-Key`, and JSON `{ "query": "..." }`)
   - `POST /admin/refresh` (rebuilds knowledge-base index; requires `X-Admin-Token` when `ISM_METRICS_ADMIN_TOKEN` is set)

## Production integration knobs
- Optional open-source integration:
  - `ISM_NLP_BACKEND` (`auto`/`spacy`/`nltk`/`basic`) to control tokenization/normalization for intent classification.
  - `ISM_OPEN_SOURCE_MARKET_DATA_ENABLED` to enable `yfinance` + `nsepython` connector attempts before existing connectors.
  - `ISM_OPEN_SOURCE_SYMBOLS` as comma-separated symbols for instrument master bootstrap (example: `RELIANCE.NS,INFY.NS,TCS.NS`).
- Embedding and reranking:
  - `ISM_EMBEDDING_LOCAL_MODEL` — SentenceTransformer model for local semantic embeddings (e.g. `all-MiniLM-L6-v2`); falls back to hash embeddings if `sentence-transformers` is not installed.
  - `ISM_RERANKER_LOCAL_MODEL_PATH` — path to a persisted `MLReranker` weights JSON file.
- Nightly refresh:
  - `ISM_NIGHTLY_REFRESH_ENABLED` (`true`/`false`) — enable the nightly knowledge-base refresh daemon.
  - `ISM_NIGHTLY_REFRESH_HOUR_UTC` — UTC hour at which the daemon fires (integer, default `2`).
- LoRA fine-tuning:
  - `ISM_TRAINING_BASE_MODEL`, `ISM_TRAINING_LORA_RANK`, `ISM_TRAINING_LORA_ALPHA`, `ISM_TRAINING_DATA_PATH`, `ISM_TRAINING_OUTPUT_PATH`
- Managed providers:
  - `ISM_EMBEDDING_PROVIDER`, `ISM_EMBEDDING_MODEL`, `ISM_EMBEDDING_ENDPOINT`, `ISM_EMBEDDING_API_KEY`
  - `ISM_RERANKER_PROVIDER`, `ISM_RERANKER_MODEL`, `ISM_RERANKER_ENDPOINT`, `ISM_RERANKER_API_KEY`
  - `ISM_MODEL_PROVIDER`, `ISM_MODEL_NAME`, `ISM_MODEL_ENDPOINT`, `ISM_MODEL_API_KEY`
- Automated rollout ingestion:
  - `ISM_ROLLOUT_INPUTS_ENDPOINT`, `ISM_ROLLOUT_INPUTS_API_KEY`, `ISM_ROLLOUT_INPUTS_TIMEOUT_SECONDS`
  - Use `ReleaseRegistry.automate_rollout_from_endpoint(...)` for remote benchmark/canary gate ingestion.
- Dashboards and alerts:
  - `ChatService` now emits `slo.alert` and `slo.alert_cleared` events based on `p95_latency_ms`, `p99_latency_ms`, `failure_rate`, and `error_budget_remaining`.
  - `evaluate_sre_readiness` maps alert outputs to the runbook at `docs/runbooks/chat_service_incident.md`.

## Integration contract for external chat boxes
- Use `ChatService` with tenant registration (`register_tenant`) to enforce per-tenant API keys.
- When tenant auth is configured, unregistered tenants are rejected (`unauthorized`).
- Requests with empty/whitespace-only queries are rejected (`bad_request`).
- Use `ChatApi` for stable API methods:
  - `health()` for liveness + contract version
  - `metrics()` for operational observability
  - `query(ApiRequest)` for authenticated tenant-scoped querying
  - `refresh()` to trigger an immediate knowledge-base index rebuild

## Production deployment topology
- Android app remains thin and calls backend HTTP endpoints only.
- Deploy the backend container behind API gateway/WAF with TLS termination.
- Keep model/provider secrets server-side only; Android sends tenant credentials only.
- Configure health probes on `/health`, private admin metrics on `/metrics`, and autoscaling on p95/p99 latency + failure-rate signals.
- Enable audit logging (`ISM_AUDIT_LOG_PATH`) with retention (`ISM_AUDIT_RETENTION_DAYS`) for regulated environments.
- Roll out in stages: internal -> beta tenants -> canary -> GA, with rollback on release-gate/SLO breach.
