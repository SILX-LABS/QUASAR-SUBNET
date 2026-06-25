# Architecture Overview

Quasar Incentive coordinates Quasar pretraining through signed work records, encrypted assignment grants, validator verification, fragment merge, checkpoint release, and Bittensor weight publication.

## Roles

### Orchestrator

The orchestrator is subnet-operated. It owns run creation, data assignment,
presigned grants, live fragment pull requests, validator-job dispatch,
accepted live-fragment merge, synced-fragment publication, and checkpoint
release.

### Miner

A miner trains assigned Quasar work. It verifies the orchestrator signature,
decrypts its assignment grant, downloads only the artifacts it was granted,
trains, answers live fragment pull requests, applies synced fragments, uploads
audit artifacts, and signs a receipt.

### Validator

A validator verifies live fragment claims, miner receipts, and artifacts. It
checks signatures, request binding, hashes, fragment metadata, GPU proof,
checkpoint lineage, and independent Quasar evaluation. It writes signed verdicts
and publishes weights from accepted live merge work.

## Trust Model

- The orchestrator signs training and validation manifests.
- Miners trust only orchestrator-signed training jobs.
- Validators trust only orchestrator-signed validation jobs.
- Validators accept only miner-signed live claims and receipts from the assigned hotkey.
- Miners and external validators do not receive operator bucket credentials.
- Private artifacts move through encrypted presigned grants.
- Checkpoint merge is controlled by the subnet operator’s merge policy.

## Live Training Path

The hot path is live fragment sync:

1. The syncer requests `fragment_id = global_step % 24`.
2. The request freezes the previous syncer-owned absolute fragment state.
3. The miner writes a signed live claim for its current absolute fragment state.
4. Validators verify the claim, hash, tensor contract, frozen previous state,
   and independent eval quality.
5. The orchestrator merges only validator-approved live claims.
6. The orchestrator publishes the updated absolute fragment state back to
   miners and advances the sync step.

This is the reward-bearing path. Miner-reported TPS, loss, and GPU details are
telemetry and dashboard signals, not payout authority.

## Training Artifacts And Receipts

Training leases also produce audit artifacts. A miner uploads:

- `fragment_update.safetensors`
- `fragment_manifest.json`
- `metrics.json`
- `gpu_proof.json` for multi-GPU jobs
- a signed receipt

The fragment manifest records the run, job, miner hotkey, fragment id/count,
base checkpoint, trained tokens, local steps, tensor metadata, and hashes.
These receipts are telemetry and audit records. Late receipts do not reopen old
live merges after the syncer has advanced.

## Validation

Validator hard checks include:

- orchestrator manifest signature,
- miner receipt signature,
- assigned-hotkey match,
- output digest match,
- fragment id/count and checkpoint lineage,
- tensor name, shape, dtype, hash, and finite-value checks,
- required GPU proof for multi-GPU jobs,
- claimed-token bounds.

Validator quality checks are run on the validator side, not the miner side. The
validator derives the trusted delta from the frozen previous syncer fragment
state and the claimed learner fragment state, applies it to a temporary Quasar
model, and evaluates assigned and heldout data before writing a verdict.

## Merge And Release

The orchestrator merges accepted validator-approved live work according to the
operator merge policy. Accepted work is weighted by capped trained tokens and
local steps, with validator quality applied by verdict. Merge state is stored as
absolute fragment state. Checkpoint release materializes an updated Quasar
checkpoint for future jobs only after accepted live events cover all 24
fragments since the last release.

## Scoring And Weights

Scores come from accepted live merge events. Failed, stale, replayed, partial,
or mismatched claims receive zero. Scores are normalized by registered hotkey,
and validators publish Bittensor weights from signed score windows.

## Bucket Objects

The bucket stores:

- current run metadata,
- checkpoint manifests and archives,
- data shard manifests and token files,
- training queues and signed job manifests,
- encrypted assignment grants,
- miner artifacts and receipts,
- validation queues and verdicts,
- merge state,
- released checkpoint manifests,
- score windows and weight publish state.

Bucket object names are implementation details. Operators should use the role CLIs and scripts instead of editing bucket objects directly.
