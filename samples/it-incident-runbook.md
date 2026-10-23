# Payments Platform Incident Runbook (Sample)

This runbook covers the error codes raised by the Payments Platform and the first steps on-call engineers should take. Codes are grouped by service. Always post in #payments-incidents before making production changes.

## Gateway service (ERR-40xx)

### ERR-4012: Gateway timeout
The upstream card processor did not respond within 8 seconds. Retries are automatic, with up to 3 attempts and exponential backoff. If the error rate stays above 2% for 10 minutes, page the Gateway on-call and switch traffic to the secondary processor with the `processor-failover` flag.

### ERR-4021: OAuth token refresh failed
The service could not refresh its OAuth client credentials with the processor. This usually means the client secret was rotated without updating the vault entry `payments/gateway/oauth`. Re-sync the secret from the processor dashboard and restart the gateway pods. Do not fail over; the secondary processor uses the same credentials.

### ERR-4027: Duplicate idempotency key
A request reused an idempotency key with a different payload. This is a client bug, not an outage. Return the original response and open a ticket for the calling team.

## Ledger service (ERR-50xx)

### ERR-5003: Ledger write conflict
Two writes tried to update the same account balance at once. The ledger retries once automatically. Repeated ERR-5003 errors on a single account point to a stuck reconciliation job; pause the job named `nightly-reconcile` and escalate to the Ledger team.

### ERR-5030: Ledger replica lag
Read replicas are more than 30 seconds behind the primary. Balance reads may be stale. Route balance reads to the primary with the `ledger-read-primary` flag until lag drops under 5 seconds.

## Escalation

- Severity 1 (payments down for all customers): page the incident commander immediately; status page update within 15 minutes.
- Severity 2 (degraded for some customers): page the owning team; status page update within 30 minutes.
- Severity 3 (no customer impact): open a ticket; no status page update.

The incident commander rotation is managed in PagerDuty under the schedule "Payments IC". Post-incident reviews are due within 5 business days for Severity 1 and 2 incidents.
