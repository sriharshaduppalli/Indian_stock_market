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
- Enforce tenant-level API key authentication.
- Keep per-tenant rate limits enabled.
- Maintain retry + circuit-breaker behavior with degraded fallback path.

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
