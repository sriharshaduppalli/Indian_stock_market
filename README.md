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
  - Grounded response composition with source citations
  - Production-style response metadata (confidence, citations, disclaimer, safety flag)
  - Prediction-intent path with live-impact factor guidance
  - Deterministic calculation support for common return/CAGR prompts
  - SEBI-aligned safety checks, prompt-injection defenses, and policy audit logs
  - Enterprise data layer scaffold with source hierarchy, validation, refresh, and lineage metadata
  - Android-ready chat serving scaffold (stable contract, cache, rate-limit, retries, circuit-breaker, fallback)
  - Evaluation release-gate scaffold for factuality, groundedness, hallucination, routing, and safety metrics
  - Versioned release registry + rollback target helper
  - Daily continual-learning feedback hook
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

## Next steps for production-grade accuracy
- Connect managed embedding/reranker/model providers for full production inference quality
- Enable automated benchmark/canary ingestion jobs feeding `ReleaseRegistry.automate_rollout_from_inputs`
- Wire dashboards/alerts to `p95_latency_ms`, `p99_latency_ms`, and `error_budget_remaining`

## Integration contract for external chat boxes
- Use `ChatService` with tenant registration (`register_tenant`) to enforce per-tenant API keys.
- When tenant auth is configured, unregistered tenants are rejected (`unauthorized`).
- Requests with empty/whitespace-only queries are rejected (`bad_request`).
- Use `ChatApi` for stable API methods:
  - `health()` for liveness + contract version
  - `metrics()` for operational observability
  - `query(ApiRequest)` for authenticated tenant-scoped querying
