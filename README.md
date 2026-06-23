# Quasar Training

Quasar Training is the production training system for continuing Quasar model
pretraining on Bittensor. It coordinates independent miners, validators, and a
subnet-operated orchestrator so real training work can be assigned, verified,
merged, and converted into subnet weights.

Our goal is to complete more than **10T additional pretraining tokens** on
Quasar Preview through the largest decentralized model-training run we can make
practical, verifiable, and repeatable.

The current base model is **Quasar Preview**:

- Model source: [`silx-ai/Quasar-Preview`](https://huggingface.co/silx-ai/Quasar-Preview)
- Architecture: Quasar MoE
- Revision: `main`
- Mainnet subnet: `netuid 24`
- Network: `finney`

The orchestrator starts from a published Quasar Preview checkpoint manifest,
splits the trainable model tensors into 24 fragments, assigns token shards to
miners, merges accepted fragment work, and periodically releases updated Quasar
checkpoints for future miners and restarts.

## System Roles

- **Miner:** trains assigned Quasar data and uploads signed training artifacts.
- **Validator:** verifies miner work, writes signed verdicts, and publishes weights.
- **Orchestrator:** subnet-operated service that assigns work, merges accepted updates, and releases checkpoints.

Most participants run either a miner or a validator. The orchestrator is run by the subnet operator.

## Live Training Flow

```text
orchestrator publishes the active run and current Quasar checkpoint
miners discover the run, download checkpoint/data through grants, and train
miners report learner progress and answer live fragment pull requests
orchestrator merges accepted fragment states and publishes updated fragments
validators verify receipts/artifacts and write signed verdicts
validators publish Bittensor weights from accepted merged work
orchestrator releases full checkpoints from absolute fragment states
```

Miners do not evaluate themselves. Validators do not assign miner work, merge updates, or release checkpoints. Miners and validators use signed manifests plus scoped grants, not broad operator credentials.

## Model And Checkpoints

Quasar Training works on Quasar Preview checkpoints, not placeholder smoke
models. A live checkpoint must include:

- the Quasar checkpoint archive or safetensors weights,
- tokenizer/config files required to load the model,
- `quasar_parameter_contract.json` or equivalent checkpoint metadata listing
  the trainable tensor names.

The parameter contract is required because the orchestrator/syncer owns global
fragment state and must split checkpoint tensors into fragments without importing
the training model code. Miners and validators load Quasar for training and
evaluation; the orchestrator coordinates tensor artifacts and checkpoint release.

Checkpoint release is assembled strictly from the latest absolute fragment
states. Delta artifacts are kept for merge/debug visibility, but released
checkpoints are built from absolute fragment state plus the base checkpoint tree.

## Mainnet Defaults

```bash
export QUASAR_NETWORK=finney
export QUASAR_NETUID=24
export QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
export QUASAR_S3_REGION=us-east-1
export QUASAR_S3_ANONYMOUS=true
```

Leave `QUASAR_RUN_ID` unset for normal mining and validation. Miners and
validators discover the active run from the public current-run metadata.

## Install

Clone the repo on the machine, then run the script for your role:

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET.git /workspace/quasar-incentive
cd /workspace/quasar-incentive
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

Validators publish weights from accepted work. If there is no accepted work to
assign yet, the validator publishes self-fallback to its own registered
validator hotkey so validator chain state stays live.

Validators do not merge model updates. Other validators only validate assigned
work, write verdicts, summarize accepted merge events, and set weights.

## Access Model

- Miners and external validators use public run metadata plus encrypted presigned grants.
- Miners verify orchestrator-signed jobs before training.
- Validators verify orchestrator-signed validation jobs and miner-signed receipts.
- Private training and validation artifacts are scoped to the assigned job.
- Operator bucket write credentials stay only on orchestrator-controlled machines.

## Development

```bash
python3 -m pip install -e ".[prod,dev]"
PYTHONPYCACHEPREFIX=/tmp/quasar-pycache pytest -q
```
