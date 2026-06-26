# Validator Guide

This guide is for running a Quasar validator worker on mainnet.

## What A Validator Does

A validator:

1. Discovers the active run.
2. Polls validation jobs assigned to its hotkey.
3. Verifies the orchestrator signature.
4. Decrypts its validation grant.
5. Downloads live fragment claims, receipt records, and artifact inputs through presigned grants.
6. Verifies miner signatures, hashes, fragment metadata, GPU proof, and checkpoint lineage.
7. Verifies live request binding to the frozen previous syncer fragment state.
8. Runs independent Quasar evaluation.
9. Writes a signed verdict bound to the exact claim and fragment hash.
10. Scores accepted live merge events and publishes Bittensor weights.

The validator does not schedule miner work, merge updates, or release checkpoints.
Only the orchestrator/syncer merges accepted work. Validators validate assigned
jobs, write verdicts, score accepted merge events, and publish Bittensor weights.

Live fragment verdicts are bound to the exact claim digest, fragment state URI
and hash, and frozen previous fragment state URI and hash. This prevents a
response from validating one artifact and merging another artifact under the
same request tuple.

The validator should treat live sync requests from run state as the active work
queue. Historical claims and expired grant records are compatibility/audit data;
they must not starve validation of current grant-backed live claims.

## Quick Start

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET.git /workspace/quasar-incentive
cd /workspace/quasar-incentive
bash scripts/install_validator.sh /workspace/quasar-incentive
```

Create or import a normal Bittensor validator wallet. Validators use the
validator hotkey identity. They do not use miner ED25519 assignment keys.

Create `.env`:

```bash
cat > .env <<'EOF'
QUASAR_NETWORK=finney
QUASAR_NETUID=24
QUASAR_WALLET_PATH=~/.bittensor/wallets
QUASAR_WALLET_NAME=<validator-wallet>
QUASAR_HOTKEY_NAME=<validator-hotkey>
QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
QUASAR_S3_REGION=us-east-1
QUASAR_S3_ANONYMOUS=true
QUASAR_OWNER_IDENTITY=5GE25P2qGpGmjzGipqezZckMvyR2mpcsJS387bbcpitNSfm5
QUASAR_WEIGHT_REFRESH_SEC=1800
EOF
```

Do not set `QUASAR_RUN_ID`. The validator discovers the active run automatically.

External validators should not use operator credentials. Validation artifacts are delivered through encrypted presigned grants.

Run preflight checks:

```bash
source .venv/bin/activate
set -a; source .env; set +a
quasar-incentive validator wallet-check
quasar-incentive validator chain-info
quasar-incentive current-run
```

The validator hotkey must be registered on the subnet for `set_weights` to succeed.

## Run With PM2

```bash
pm2 delete quasar-validator || true
pm2 start /usr/bin/bash --name quasar-validator -- -lc 'cd /workspace/quasar-incentive && bash scripts/run_validator.sh'
pm2 logs quasar-validator
```

The loop:

- consumes assigned validation jobs,
- writes verdicts,
- verifies live fragment claims and receipt telemetry,
- summarizes accepted live merge events,
- publishes changed positive weights immediately,
- retries failed or rate-limited `set_weights` with backoff,
- refreshes unchanged positive weights periodically,
- keeps running.

The operator validation dispatcher assigns validation jobs. External validators run only the validator loop above.

## Weight Publication

Validators publish weights only after accepted live merge work exists. The
validator does not publish fake or random miner weights before there is positive
accepted score. If there is no accepted score to assign, the validator must
publish self-fallback to its own registered validator hotkey. In the current
operator deployment that is validator UID `155`. External validators use their
own registered validator UID for the same self-fallback rule.

Default behavior:

- new accepted scores publish immediately,
- Bittensor `weights_rate_limit` is checked before submission,
- rate-limited submissions wait until the required block instead of hammering chain,
- failed submissions retry with backoff,
- empty accepted-score windows publish self-fallback to the validator hotkey,
- unchanged positive scores refresh every `QUASAR_WEIGHT_REFRESH_SEC`,
- default refresh interval is `1800` seconds.

Scores are summarized from accepted live merge events. The default score window
uses recent merge events with decay, so miners must continue contributing valid
live fragments to keep weight. Miner-reported TPS, loss, GPU names, and final
receipt claims are not direct payout inputs.

## Troubleshooting

If the validator is idle:

- confirm `QUASAR_OWNER_IDENTITY` is the orchestrator hotkey,
- confirm the hotkey is registered on the netuid,
- confirm the validator hotkey has been enabled by the subnet operator,
- confirm the active run exists,
- confirm validation jobs are being dispatched.

If `set_weights` fails:

- confirm the validator hotkey is registered,
- confirm the validator has a validator permit if the subnet requires it,
- check the chain rejection reason in logs,
- wait for Bittensor rate limits if logs show `reason=ratelimited`,
- confirm there is at least one accepted score window.

Useful log filters:

```bash
pm2 logs quasar-validator --lines 200
tail -n 300 /root/.pm2/logs/quasar-validator-out.log | grep -E 'validator_pass|weight_publish|ratelimited|submitted|score_fingerprint'
tail -n 120 /root/.pm2/logs/quasar-validator-error.log
```
