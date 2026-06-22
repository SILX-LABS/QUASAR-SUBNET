# Orchestrator Guide

The orchestrator is subnet-operated. It is the only role that should create runs, assign miner jobs, mint presigned grants, merge accepted updates, and release checkpoints.

Do not run the orchestrator on third-party miner or validator machines.

## What The Orchestrator Does

1. Publishes the active run.
2. Publishes or references the current Quasar checkpoint.
3. Maintains the approved data shard catalog.
4. Discovers registered miner workers.
5. Emits signed training jobs.
6. Creates encrypted assignment grants with presigned artifact access.
7. Dispatches validator jobs to configured validation targets.
8. Reconciles completed, skipped, and expired work.
9. Merges accepted fragment updates.
10. Releases updated checkpoints.

## Install

```bash
cd /workspace/quasar-incentive
bash scripts/install_orchestrator.sh /workspace/quasar-incentive
```

## Environment

The orchestrator needs:

- orchestrator wallet,
- bucket write credentials,
- Quasar model runtime paths,
- a starting checkpoint manifest,
- validator hotkeys that should receive validation jobs.

```bash
export QUASAR_NETWORK=<network>
export QUASAR_NETUID=<netuid>
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<orchestrator-wallet>
export QUASAR_HOTKEY_NAME=<orchestrator-hotkey>
export QUASAR_S3_BUCKET=<bucket>
export QUASAR_S3_REGION=<region>
export QUASAR_CHECKPOINT_MANIFEST_URI=<checkpoint-manifest-uri>
```

Configure bucket credentials through the host’s normal cloud credential mechanism. Keep those credentials only on operator-controlled machines.

## Readiness Checks

```bash
quasar-incentive orchestrator s3-check
quasar-incentive orchestrator wallet-check
quasar-incentive orchestrator chain-info
```

## Data And Checkpoint

The orchestrator uses the prepared shard catalog by default. Approved sources are defined in `incentive/data/sources.py`.

Typical production settings:

```bash
export QUASAR_DYNAMIC_SHARD_CATALOG=1
export QUASAR_AUTO_PREPARE_SHARDS=1
export QUASAR_DATA_STAGES=pretrain,midtrain
```

To publish an initial checkpoint, provide a real model checkpoint or archive:

```bash
quasar-incentive orchestrator publish-checkpoint \
  --global-step 0 \
  --weights-path <checkpoint-archive-or-weights-file>
```

Do not use a smoke/placeholder checkpoint for a live run; validators need real
initial weights for fragment sync.

To prepare shards manually:

```bash
quasar-incentive orchestrator prepare-shards \
  --source all \
  --stage pretrain \
  --tokens-per-shard <tokens-per-shard> \
  --sequence-length <sequence-length> \
  --max-shards <count>
```

## Run

```bash
bash scripts/run_orchestrator.sh
```

The loop keeps running. It emits miner jobs, dispatches validator jobs, finalizes rounds, merges accepted work, releases checkpoints, and advances the active run state.

## W&B Telemetry

Orchestrator W&B is optional. If `WANDB_API_KEY` is present, the orchestrator
logs run, round, miner discovery, validation-job, live-sync, merge, release, and
waiting-state events.

```bash
export WANDB_API_KEY=<wandb-token>
export QUASAR_WANDB_PROJECT=quasar-incentive
export QUASAR_ORCHESTRATOR_WANDB_RUN_NAME=<run-name>
```

If no W&B key is set, the orchestrator keeps running normally without telemetry.

## Validators

Validators do not self-enroll into production validation. Set the validator hotkeys that should receive validation jobs:

```bash
export QUASAR_VALIDATOR_HOTKEYS=<validator-hotkey-a>,<validator-hotkey-b>
```

The validation dispatcher can run beside the orchestrator:

```bash
bash scripts/run_validation_jobs.sh
```

External validators run only `scripts/run_validator.sh`.

## Merge And Release

Production defaults should keep automatic finalization enabled:

```bash
export QUASAR_AUTO_MERGE_RELEASE=1
export QUASAR_AUTO_RELEASE_CHECKPOINTS=1
```

The orchestrator merges only accepted validator verdicts. When a round finalizes, it records merge state, queues checkpoint release, updates active checkpoint metadata, and emits future work from the updated state.

## Safety Rules

- Only the operator-controlled orchestrator has bucket write credentials.
- Only configured validators receive validation jobs.
- Checkpoint merge is controlled by the operator’s merge policy.
- Public heartbeats do not make a miner a validator.
- Miners and validators receive artifact access through encrypted presigned grants.
- Keep `.env`, wallets, hotkeys, cloud keys, and generated run artifacts out of git.
