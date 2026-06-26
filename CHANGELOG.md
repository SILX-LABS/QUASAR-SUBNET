# Changelog

## 2026-06-26

### Production Documentation Cleanup

- Clarified that accepted live merge events, not final receipts or miner
  self-reports, are the authoritative scoring path.
- Documented validator score decay and the recent-event scoring window.
- Documented checkpoint release as operational recovery infrastructure for late
  joiners and restarts, while live fragment sync remains the training hot path.
- Updated miner guidance for latest-code live claims, automatic run discovery,
  OOM recovery, and avoiding duplicate hotkey use across machines.
- Updated validator guidance for active live request validation, self-fallback,
  and rate-limited weight refresh behavior.

## 2026-06-25

### Live Sync And Merge

- Made live fragment sync the authoritative training merge path.
- Added signed live fragment claims for pull-fragment responses.
- Bound live claims to run id, request id, learner id, miner hotkey, worker id,
  fragment id/count, global step, local step, counters, fragment state hash, and
  the frozen previous syncer-owned fragment state.
- Froze previous `Theta_p` URI and hash per live sync request so validators and
  merge use the same reference state.
- Required live merge to use validator-approved live responses instead of raw
  learner responses.
- Kept RDA and Nesterov outer merge behavior unchanged.

### Validator Security

- Added live fragment verdicts that are signed by validator hotkey.
- Bound verdicts to the exact claim digest, fragment state URI/hash, and frozen
  previous fragment state URI/hash.
- Added strict rejection for wrong request id, wrong fragment id/count, stale
  response, bad hash, missing/extra tensor, shape mismatch, non-finite tensors,
  and bad signatures.
- Validator derives trusted deltas from previous syncer state and learner state;
  miner-supplied deltas are not trusted.

### Scoring And Weights

- Shifted rewards to accepted live merge events.
- Kept miner-reported TPS, GPU info, and loss as telemetry only.
- Added accepted-work decay so old score history does not dominate current
  compute indefinitely.
- Added resource sanity penalties for impossible or suspicious throughput claims.
- Kept validator self-fallback behavior for empty accepted-score windows.

### Checkpoint Release

- Release now waits for accepted live coverage of all 24 fragments.
- Release assembles checkpoints from absolute fragment states only.
- Added release processing before new work emission so checkpoint release is not
  starved by continuous scheduling.
- Added duplicate same-cycle live release suppression after a checkpoint has
  already been published.

### Dashboard And Docs

- Updated dashboard counters to read accepted live merge events.
- Updated miner, validator, orchestrator, and architecture docs for the live
  validator-gated merge flow.
- Added tests for live claims, artifact-bound verdicts, release ordering,
  scoring behavior, and dashboard live counters.
