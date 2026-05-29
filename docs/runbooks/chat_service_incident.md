# Chat Service Incident Runbook

1. Verify current p95/p99 latency, failure rate, and error-budget burn from service metrics.
2. Check canary status and compare tenant-specific error spikes.
3. Trigger rollback using latest `ReleaseRegistry.rollback_target()` when SLO breach persists.
4. Pause auto-promotion and drain traffic to safe mode if degradation continues.
5. Record incident timeline and run rollback drill outcomes for postmortem.
