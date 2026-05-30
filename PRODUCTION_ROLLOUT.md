# Production Rollout Checklist

## Release gates
- Run benchmark gate (factuality, calculations, groundedness, hallucination, safety, routing).
- Run operational gate (uptime, latency, cost, blocked ratio, failure rate).
- Run regression gate (`passes_regression_gate`) before promotion.

## Data readiness blockers
- Block factual responses when data readiness report is not ready.
- Treat stale feeds, partial feeds, and incomplete datasets as release blockers.
- Require lineage metadata for all enterprise datasets.

## Serving readiness
- Run network endpoint using `python -m indian_stock_llm.http_server` to expose:
  - `GET /health`
  - `GET /metrics` (admin/internal)
  - `POST /query` (tenant-scoped auth and request validation)
- Enforce tenant-level API key authentication.
- Support key rotation by storing multiple active keys per tenant.
- Keep per-tenant rate limits enabled.
- Reject empty/whitespace queries and enforce max query length controls.
- Maintain retry + circuit-breaker behavior with degraded fallback path.

## Deployment topology and perimeter security
- Containerize backend and deploy behind API gateway + WAF + TLS termination.
- Route Android traffic only to backend query endpoint; never expose provider/model secrets to clients.
- Use secure secret management for connector/model/monitoring API keys.
- Apply gateway abuse controls: rate limits, bot checks, and IP/device anomaly checks.

## Canary and GA rollout
- Run canary approval with `assess_canary`.
- Monitor canary error rate before full rollout.
- Promote to GA only when canary and release gates pass.
- Roll back to `rollback_target` when rollout criteria fail.
- For automated external benchmark/canary ingestion, use `automate_rollout_from_endpoint` with endpoint/API key inputs.

## Incident and recovery
- Use service metrics + policy logs for incident triage.
- Keep release registry history current for rollback decisions.
- Validate data-source fallback behavior during failover drills.
- Route `slo.alert`/`slo.alert_cleared` monitoring events to on-call dashboards.
- Drill incident runbook + rollback path regularly.

## Data reliability and SLA controls
- Enable live connectors in production (`ISM_LIVE_CONNECTORS_ENABLED=true`) and configure `ISM_*_CONNECTOR_URL` endpoints.
- Treat stale, partial, and incomplete dataset states as release blockers.
- Monitor connector health + lineage for each dataset and alert on freshness SLA misses.
- Define fallback actions for connector failures, model timeout, and degraded mode responses.

## Compliance and governance
- Run release-gate checks using `ProductionAcceptanceCriteria` and block promotion when thresholds fail.
- Capture end-to-end audit events and retain logs per regulated retention policy.
- Execute red-team and policy validation for prompt injection and unsafe financial-advice paths.
- Complete legal/compliance sign-off for disclaimers, advice boundaries, and prohibited-use enforcement.

## Pre-go-live validation
- Functional: full regression + scenario tests for Indian market intents and safety.
- Non-functional: load/stress testing for Android traffic SLOs.
- Security: penetration/API abuse tests plus secret/key-rotation checks.
- Launch strategy: internal -> beta tenant -> canary -> full rollout with explicit rollback triggers.
- Immediate next step: run staging pilot using the HTTP wrapper with Android app before GA.
