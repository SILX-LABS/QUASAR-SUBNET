# Architecture Overview

Quasar Incentive coordinates Quasar pretraining through signed work records, encrypted assignment grants, validator verification, fragment merge, checkpoint release, and Bittensor weight publication.

## Roles

### Orchestrator

The orchestrator is subnet-operated. It owns run creation, data assignment, presigned grants, validator-job dispatch, accepted-update merge, and checkpoint release.

### Miner

A miner trains assigned Quasar work. It verifies the orchestrator signature, decrypts its assignment grant, downloads only the artifacts it was granted, trains, uploads fragment artifacts, and signs a receipt.

### Validator

A validator verifies miner receipts and artifacts. It checks signatures, hashes, fragment metadata, GPU proof, checkpoint lineage, and independent Quasar evaluation. It writes signed verdicts and publishes weights from accepted work.

## Trust Model

- The orchestrator signs training and validation manifests.
- Miners trust only orchestrator-signed training jobs.
- Validators trust only orchestrator-signed validation jobs.
- Validators accept only miner-signed receipts from the assigned hotkey.
- Miners and external validators do not receive operator bucket credentials.
- Private artifacts move through encrypted presigned grants.
- Checkpoint merge is controlled by the subnet operator’s merge policy.

## Training Artifacts

Training jobs request one model fragment. A miner uploads:

- `fragment_update.safetensors`
- `fragment_manifest.json`
- `metrics.json`
- `gpu_proof.json` for multi-GPU jobs
- a signed receipt

The fragment manifest records the run, job, miner hotkey, fragment id/count, base checkpoint, trained tokens, local steps, tensor metadata, and hashes.

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

Validator quality checks are run on the validator side, not the miner side. The validator applies the fragment to a temporary Quasar model and evaluates assigned and heldout data before writing a verdict.

## Merge And Release

The orchestrator merges accepted validator-approved work according to the operator merge policy. Accepted work is weighted by trained tokens and training throughput. Merge state is stored per fragment, then checkpoint release materializes an updated Quasar checkpoint for future jobs.

## Scoring And Weights

Scores come from accepted merged work. Failed work is penalized. Scores are normalized by registered hotkey, and validators publish Bittensor weights from signed score windows.

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
