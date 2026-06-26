# Miner Guide

This guide is for running a Quasar miner.

## What A Miner Does

A miner:

1. Publishes a miner heartbeat.
2. Discovers the active run.
3. Polls for jobs assigned to its hotkey.
4. Verifies the orchestrator signature.
5. Decrypts its assignment grant.
6. Downloads checkpoint and data through presigned grants.
7. Trains Quasar.
8. Answers live fragment pull requests while training.
9. Applies synced fragments published by the orchestrator.
10. Uploads training artifacts and a signed receipt for audit.
11. Continues polling for more work.

The miner does not evaluate itself and does not need operator credentials.
Private artifacts are downloaded and uploaded through scoped grants. Miner
self-reported TPS, GPU details, and loss are telemetry only; rewards come from
validator-approved live merge events.

Current protocol requires the latest miner code. Live fragment responses are
uploaded as signed live claims and bound to the syncer's request id, fragment id,
fragment hash, and previous global fragment state. Old miners may still poll or
train, but they will not be reliable for the validator-gated live merge path.

## Install

```bash
cd /workspace/quasar-incentive
bash scripts/install_miner.sh /workspace/quasar-incentive
```

## Wallet

Use a registered native ED25519 hotkey:

```bash
quasar-generate-ed25519-hotkey \
  --wallet-name <miner-wallet> \
  --hotkey <miner-hotkey>
```

Register the hotkey on the target subnet before mining.

## Environment

For the current Quasar mainnet subnet:

```bash
export QUASAR_NETWORK=finney
export QUASAR_NETUID=24
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<miner-wallet>
export QUASAR_HOTKEY_NAME=<miner-hotkey>
export QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
export QUASAR_S3_REGION=us-east-1
export QUASAR_S3_ANONYMOUS=true
```

Do not set `QUASAR_RUN_ID` for normal mining. The miner discovers the active run automatically.
When switching machines or restarting after an old run, keep the wallet/hotkey
the same only if you are intentionally moving that miner. Do not run the same
hotkey on two machines at once.

## Run

```bash
quasar-incentive miner run \
  --worker-id "$(hostname)-0" \
  --owner-identity 5GE25P2qGpGmjzGipqezZckMvyR2mpcsJS387bbcpitNSfm5
```

For a multi-GPU node, the default launch mode advertises the node as one grouped worker. To select devices:

```bash
export QUASAR_MINER_DEVICES=0,1,2,3
quasar-incentive miner run \
  --worker-id "$(hostname)-0" \
  --owner-identity 5GE25P2qGpGmjzGipqezZckMvyR2mpcsJS387bbcpitNSfm5
```

The miner publishes approximate datacenter-area location automatically from the
server public IP. No manual location config is required.

## Training And Sync Behavior

Quasar mining is not winner-take-all. The orchestrator pulls live fragments from
active learners while they train. A miner earns score only when its live fragment
claim is validated and accepted into a live merge event. Final receipts are kept
for audit and telemetry, but accepted live fragment merges are the reward path.

The miner may receive synced fragments while training. Applying a synced
fragment resets only that fragment's local counters; the learner keeps training
instead of waiting for a full checkpoint release.

## Expected Logs

Useful events:

- `quasar_miner_training_start`
- `quasar_parallel_mode`
- `quasar_miner_mesh_pull_response_uploaded`
- `quasar_miner_mesh_sync_received`
- `quasar_miner_mesh_sync_applied`
- `quasar_miner_upload_start`
- `quasar_miner_receipt_upload_done`
- `quasar_miner_job_skipped`

`grant expired` during idle mailbox polling usually means an old scoped grant
expired. If the miner is actively training, do not stop it for that line alone.
New live-control grants are issued by the orchestrator as work advances.

## Troubleshooting

If the miner is idle:

- confirm the hotkey is registered on the netuid,
- confirm `QUASAR_S3_ANONYMOUS=true`,
- confirm the bucket allows public current-run and queue reads,
- confirm `--owner-identity` is the orchestrator hotkey,
- confirm the orchestrator has an active run and assigned work.

If anonymous public metadata access is denied, fix the public metadata policy. Do not add operator credentials to a miner.

If a job OOMs:

- restart the miner process after GPU memory is clear,
- reduce `QUASAR_BATCH_SIZE` if you set it manually,
- keep FSDP enabled on multi-GPU systems when supported,
- do not keep retrying a configuration that repeatedly OOMs before serving live fragments.
