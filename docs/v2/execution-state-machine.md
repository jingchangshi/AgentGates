# AgentGates V2 — Execution State Machine

This document is normative: `status()` reconciles an execution into exactly the
states below, and no transition leaves a terminal state.

## States

| State | Terminal | Meaning |
|---|---|---|
| `SUBMITTED` | no | The request is durable in the store; the worker unit has been started. |
| `STARTING` | no | The worker has begun, or systemd reports the unit activating. |
| `RUNNING` | no | The worker is executing gates through the Engine. |
| `CANCELLING` | no | A cancel was requested and the unit is being stopped. |
| `FINISHED` | **yes** | The worker stopped and published a receipt. |
| `CANCELLED` | **yes** | The run was cancelled by an operator, before or during execution. |
| `INTERRUPTED` | **yes** | The run's outcome cannot be proven: the worker died, the machine rebooted, or no receipt was published. |

## The one rule that matters

`FINISHED` is **not** `PASS`.

`FINISHED` describes the worker. `gate_status` describes the gates. A finished
execution whose gates FAIL is the normal, expected outcome of a broken tree, and
it is reported with `state: FINISHED, gate_status: FAIL`. A completion gate reads
`gate_status` (and, more strictly, re-derives it via `verify_requirements`), never
`state`.

## Transitions

```
           submit
             |
        SUBMITTED --------- current state after submit returns
             |
             |  worker writes / systemd reports activating
             v
         STARTING
             |
             v
        RUNNING ---+
             |      | cancel()
             |      v
             |  CANCELLING
             |      |
             |      +--> CANCELLED
             |
             +--> FINISHED   (receipt published; gate_status set separately)
             +--> INTERRUPTED (worker died / reboot / no receipt)
```

`CANCELLED` and `INTERRUPTED` are also reachable directly from `SUBMITTED`/
`STARTING`: a cancel before the worker starts is `CANCELLED`, and a worker that
never reports is `INTERRUPTED`.

## Reconciliation (D6)

`status()` never returns a state file verbatim. It re-derives the answer from:

1. the durable status record,
2. the systemd unit's `ActiveState`/`Result`,
3. the worker process identity — pid **and** boot id **and** a command-line
   check, because a pid can be reused after a reboot,
4. whether a receipt exists, and
5. whether that receipt *verifies*.

Only a receipt that passes verification may upgrade a record. The two crash
windows the objective names are covered explicitly:

- *Receipt written, status not updated* → the receipt is discovered and adopted,
  the execution becomes `FINISHED` with the receipt's real `gate_status`.
- *Status updated, final event not written* → the status file is authoritative;
  the missing event line is cosmetic, and `events.jsonl` is documented as
  diagnostic only.

If neither a live worker nor a verifying receipt can be proven, the execution is
`INTERRUPTED`. It is never inferred as successful from a historical file.

## Gate verdicts (unchanged from V1)

| Verdict | Meaning |
|---|---|
| `PASS` | Every observation assertion held, source and artifacts unchanged. |
| `FAIL` | A test genuinely failed. |
| `BLOCKED_ENV` | The environment prevented the test from running (unconfigured toolchain, busy NPU card, blocked exit code). No assertion ran. |
| `NOT_RUN` | Never started, e.g. a dependency did not pass. |
| `STALE` | Passed, but the source, configuration, adapter or artifacts changed since. |
| `INVALID_EVIDENCE` | The receipt is malformed or cannot be trusted. |

`aggregate()` is positive: a run is `PASS` only when **every** gate is literally
`PASS`. An unrecognised verdict becomes `INVALID_EVIDENCE`, never `PASS`.

## Error taxonomy

Execution state is a *lifecycle*. `APIError` is a *refusal*, and its `family`
tells a caller which layer refused (see `docs/v2/adr.md` ADR-007). `BLOCKED_ENV`
appears only as a gate verdict; it is never used as an error code.
