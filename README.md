# Quasar Incentive

Quasar Incentive coordinates distributed Quasar pretraining on Bittensor. It lets independent miners train real Quasar shards, lets validators verify submitted work, and turns accepted work into subnet weights and updated model checkpoints.

It has three roles:

- **Miner:** trains assigned Quasar data and uploads signed training artifacts.
- **Validator:** verifies miner work, writes signed verdicts, and publishes weights.
- **Orchestrator:** subnet-operated service that assigns work, merges accepted updates, and releases checkpoints.

Most participants run either a miner or a validator. The orchestrator is run by the subnet operator.

## How It Works

```text
orchestrator signs jobs and publishes the active run
miners train assigned work and sign receipts
validators verify receipts and sign verdicts
orchestrator merges accepted work and releases updated checkpoints
validators publish weights from accepted work
```

Miners do not evaluate themselves. Validators do not assign miner work. Miners and validators use signed manifests plus scoped grants, not broad operator credentials.

## Install

Clone the repo on the machine, then run the script for your role:

```bash
bash scripts/install_miner.sh /workspace/quasar-incentive
bash scripts/install_validator.sh /workspace/quasar-incentive
```

## Mainnet Miner Quick Start

Current mainnet settings:

```bash
export QUASAR_NETWORK=finney
export QUASAR_NETUID=24
export QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
export QUASAR_S3_REGION=us-east-1
export QUASAR_S3_ANONYMOUS=true
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<miner-wallet>
export QUASAR_HOTKEY_NAME=<miner-hotkey>
```

Run:

```bash
quasar-incentive miner run \
  --worker-id "$(hostname)-0" \
  --owner-identity 5GE25P2qGpGmjzGipqezZckMvyR2mpcsJS387bbcpitNSfm5
```

Leave `QUASAR_RUN_ID` unset. The miner discovers the live run automatically.

## Common Environment

Set these for the role wallet and subnet:

```bash
export QUASAR_NETWORK=<network>
export QUASAR_NETUID=<netuid>
export QUASAR_WALLET_PATH=~/.bittensor/wallets
export QUASAR_WALLET_NAME=<wallet>
export QUASAR_HOTKEY_NAME=<hotkey>
```

Miners and external validators also use the subnet-provided public bucket metadata:

```bash
export QUASAR_S3_BUCKET=<bucket>
export QUASAR_S3_REGION=<region>
export QUASAR_S3_ANONYMOUS=true
```

Do not set `QUASAR_RUN_ID` for normal mining or validation. The active run is discovered automatically.

## Run A Miner

Create or use a registered ED25519 miner hotkey, then run:

```bash
quasar-incentive miner run \
  --worker-id <worker-id> \
  --owner-identity <orchestrator-hotkey>
```

For multi-GPU miners:

```bash
export QUASAR_MINER_DEVICES=0,1,2,3
quasar-incentive miner run \
  --worker-id <worker-id> \
  --owner-identity <orchestrator-hotkey>
```

More: [docs/mining.md](docs/mining.md)

## Run A Validator

Use a registered validator hotkey and set the orchestrator identity:

```bash
export QUASAR_OWNER_IDENTITY=<orchestrator-hotkey>
bash scripts/run_validator.sh
```

More: [docs/validator.md](docs/validator.md)

Validators publish weights from accepted work. If scores are unchanged, positive
weights refresh periodically so quiet runs keep live chain state.

## Access Model

- Miners and external validators use public run metadata plus encrypted presigned grants.
- Miners verify orchestrator-signed jobs before training.
- Validators verify orchestrator-signed validation jobs and miner-signed receipts.
- Private training and validation artifacts are scoped to the assigned job.

## Development

```bash
python3 -m pip install -e ".[prod,dev]"
PYTHONPYCACHEPREFIX=/tmp/quasar-pycache pytest -q
```
