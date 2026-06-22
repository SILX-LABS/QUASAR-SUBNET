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
8. Uploads training artifacts and a signed receipt.
9. Continues polling for more work.

The miner does not evaluate itself and does not need operator credentials. Private artifacts are downloaded and uploaded through scoped grants.

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

## Expected Logs

Useful events:

- `quasar_miner_training_start`
- `quasar_parallel_mode`
- `quasar_miner_upload_start`
- `quasar_miner_receipt_upload_done`
- `quasar_miner_job_skipped`

## Troubleshooting

If the miner is idle:

- confirm the hotkey is registered on the netuid,
- confirm `QUASAR_S3_ANONYMOUS=true`,
- confirm the bucket allows public current-run and queue reads,
- confirm `--owner-identity` is the orchestrator hotkey,
- confirm the orchestrator has an active run and assigned work.

If anonymous public metadata access is denied, fix the public metadata policy. Do not add operator credentials to a miner.
