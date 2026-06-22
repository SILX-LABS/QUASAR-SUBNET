# Validator Guide

This guide is for running a Quasar validator worker.

## What A Validator Does

A validator:

1. Discovers the active run.
2. Polls validation jobs assigned to its hotkey.
3. Verifies the orchestrator signature.
4. Decrypts its validation grant.
5. Downloads receipt and artifact inputs through presigned grants.
6. Verifies miner signatures, hashes, fragment metadata, GPU proof, and checkpoint lineage.
7. Runs independent Quasar evaluation.
8. Writes a signed verdict.
9. Scores accepted work and publishes Bittensor weights.

The validator does not schedule miner work and does not release checkpoints.

## Install

```bash
cd /workspace/quasar-incentive
bash scripts/install_validator.sh /workspace/quasar-incentive
```

## Environment

```bash
export QUASAR_NETWORK=<network>
export QUASAR_NETUID=<netuid>
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<validator-wallet>
export QUASAR_HOTKEY_NAME=<validator-hotkey>
export QUASAR_S3_BUCKET=<bucket>
export QUASAR_S3_REGION=<region>
export QUASAR_S3_ANONYMOUS=true
export QUASAR_OWNER_IDENTITY=<orchestrator-hotkey>
```

Do not set `QUASAR_RUN_ID` for normal validation. The validator discovers the active run automatically.

External validators should not use operator credentials. Validation artifacts are delivered through encrypted presigned grants.

## Checks

```bash
quasar-incentive validator wallet-check
quasar-incentive validator chain-info
quasar-incentive current-run
```

The validator hotkey must be registered on the subnet for `set_weights` to succeed.

## Run

```bash
bash scripts/run_validator.sh
```

The loop:

- consumes assigned validation jobs,
- writes verdicts,
- summarizes accepted work,
- publishes changed positive weights immediately,
- retries failed or rate-limited `set_weights` with backoff,
- refreshes unchanged positive weights periodically,
- keeps running.

The operator validation dispatcher assigns validation jobs. External validators run only the validator loop above.

## Weight Publication

Validators publish weights only after accepted work exists. The validator does
not publish fake or random weights before there is positive accepted score.

Default behavior:

- new accepted scores publish immediately,
- failed or rate-limited submissions retry with backoff,
- unchanged positive scores refresh every `QUASAR_WEIGHT_REFRESH_SEC`,
- default refresh interval is `1800` seconds.

To change the refresh interval:

```bash
export QUASAR_WEIGHT_REFRESH_SEC=1800
```

## Troubleshooting

If the validator is idle:

- confirm `QUASAR_OWNER_IDENTITY` is the orchestrator hotkey,
- confirm the hotkey is registered on the netuid,
- confirm the validator hotkey has been enabled by the subnet operator,
- confirm the active run exists,
- confirm validation jobs are being dispatched.

If `set_weights` fails:

- confirm the validator hotkey is registered,
- check the chain rejection reason in logs,
- wait for Bittensor rate limits/backoff if applicable,
- confirm there is at least one accepted score window.
