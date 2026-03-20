# Complaints

Purpose: a lightweight ledger for recurring pain points in the repo, harness, cloud workflow, or evaluation process.

Use this file when:
- a problem wastes time more than once
- the fix is non-trivial or should be prioritized later
- the issue is about tooling, workflow, infrastructure, or experiment discipline rather than model quality itself

Keep entries short:
- `Date`
- `Area`
- `Complaint`
- `Impact`
- `Next fix`
- `Status`

Example:

## 2026-03-20 - Runpod Logging Split

- Area: cloud harness
- Complaint: pod smoke runs can write the useful trainer output to `logs/<RUN_ID>.txt` while the outer `nohup`/driver log stays empty, which makes progress checks and metric capture slower than they should be.
- Impact: wastes cloud time and causes false uncertainty about whether a run is healthy.
- Next fix: add one canonical launch wrapper that writes a single driver log plus explicit done/failed sentinels.
- Status: open

## 2026-03-20 - Evidence Ladder Is Too Implicit

- Area: experiment process
- Complaint: local short-proxy reruns, same-checkpoint exporter evals, and capped cloud smoke runs all answer different questions, but the repo does not state clearly which evidence outranks which.
- Impact: makes it too easy to overreact to weak signals or compare mismatched results.
- Next fix: add a short comparison ladder near the top of `AGENTS.md`.
- Status: open
