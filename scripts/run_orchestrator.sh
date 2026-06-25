#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

QUASAR_RUN_ID="${QUASAR_RUN_ID:-${RUN_ID:-}}"
export QUASAR_RUN_ID

STEPS="${ORCHESTRATOR_STEPS:-${QUASAR_ORCHESTRATOR_STEPS:-1000000}}"
TIMEOUT_SEC="${ORCHESTRATOR_TIMEOUT_SEC:-${QUASAR_ORCHESTRATOR_TIMEOUT_SEC:-31536000}}"
POLL="${ORCHESTRATOR_POLL_INTERVAL:-${QUASAR_ORCHESTRATOR_POLL_INTERVAL:-5}}"

export QUASAR_DYNAMIC_SHARD_CATALOG="${QUASAR_DYNAMIC_SHARD_CATALOG:-1}"
export QUASAR_DATA_STAGES="${QUASAR_DATA_STAGES:-pretrain,midtrain}"
export QUASAR_AUTO_PREPARE_SHARDS="${QUASAR_AUTO_PREPARE_SHARDS:-1}"
export QUASAR_AUTO_PREPARE_INTERVAL_SEC="${QUASAR_AUTO_PREPARE_INTERVAL_SEC:-1800}"
export QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD="${QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD:-67108864}"
export QUASAR_AUTO_PREPARE_MIN_TRAIN_SHARDS="${QUASAR_AUTO_PREPARE_MIN_TRAIN_SHARDS:-64}"
export QUASAR_AUTO_PREPARE_MAX_NEW_SHARDS="${QUASAR_AUTO_PREPARE_MAX_NEW_SHARDS:-16}"
export QUASAR_INDEPENDENT_QUASAR_EVAL="${QUASAR_INDEPENDENT_QUASAR_EVAL:-1}"
export QUASAR_SYNC_QUORUM="${QUASAR_SYNC_QUORUM:-1}"
export QUASAR_ROUND_MERGE_MIN_GRACE_SEC="${QUASAR_ROUND_MERGE_MIN_GRACE_SEC:-900}"
export QUASAR_ROUND_MERGE_MAX_GRACE_SEC="${QUASAR_ROUND_MERGE_MAX_GRACE_SEC:-1800}"
export QUASAR_MIN_SHARD_TOKENS="${QUASAR_MIN_SHARD_TOKENS:-${QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD}}"
export QUASAR_AUTO_MERGE_RELEASE="${QUASAR_AUTO_MERGE_RELEASE:-1}"
export QUASAR_AUTO_RELEASE_CHECKPOINTS="${QUASAR_AUTO_RELEASE_CHECKPOINTS:-1}"
export QUASAR_REQUIRE_REGISTERED_MINERS="${QUASAR_REQUIRE_REGISTERED_MINERS:-1}"
export QUASAR_JOB_MIN_GPUS="${QUASAR_JOB_MIN_GPUS:-1}"
export QUASAR_LOCAL_STEPS="${QUASAR_LOCAL_STEPS:-auto}"
export QUASAR_MAX_SEQUENCES="${QUASAR_MAX_SEQUENCES:-auto}"

ALLOW_TEST_RUN="${QUASAR_ALLOW_TEST_RUN:-0}"
is_test_run_allowed() {
  case "${ALLOW_TEST_RUN,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_production_launch() {
  if is_test_run_allowed; then
    echo "[orchestrator] test-run overrides enabled by QUASAR_ALLOW_TEST_RUN=1"
    return 0
  fi

  if [[ "${QUASAR_RUN_ID:-}" =~ (500k|smoke) ]]; then
    echo "error: refusing production orchestrator launch with test run_id=${QUASAR_RUN_ID}" >&2
    echo "set QUASAR_ALLOW_TEST_RUN=1 only for an intentional smoke run" >&2
    exit 2
  fi
  if [[ -n "${QUASAR_SHARD_MANIFEST_URIS:-${QUASAR_SHARD_MANIFEST_URI:-}}" || -n "${QUASAR_EVAL_SHARD_MANIFEST_URIS:-${QUASAR_EVAL_SHARD_MANIFEST_URI:-}}" ]]; then
    echo "error: refusing production orchestrator launch with explicit shard manifest overrides" >&2
    echo "clear QUASAR_SHARD_MANIFEST_URI(S) and QUASAR_EVAL_SHARD_MANIFEST_URI(S), or set QUASAR_ALLOW_TEST_RUN=1 for a manual test run" >&2
    exit 2
  fi
  if [[ "${QUASAR_DYNAMIC_SHARD_CATALOG}" != "1" && "${QUASAR_DYNAMIC_SHARD_CATALOG,,}" != "true" ]]; then
    echo "error: production orchestrator requires QUASAR_DYNAMIC_SHARD_CATALOG=1" >&2
    exit 2
  fi
  if [[ "${QUASAR_AUTO_PREPARE_SHARDS}" != "1" && "${QUASAR_AUTO_PREPARE_SHARDS,,}" != "true" ]]; then
    echo "error: production orchestrator requires QUASAR_AUTO_PREPARE_SHARDS=1" >&2
    exit 2
  fi
  if (( QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD < 67108864 )); then
    echo "error: production tokens_per_shard must be at least 67108864, got ${QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD}" >&2
    exit 2
  fi
  if (( QUASAR_MIN_SHARD_TOKENS < 67108864 )); then
    echo "error: production min_shard_tokens must be at least 67108864, got ${QUASAR_MIN_SHARD_TOKENS}" >&2
    exit 2
  fi
}

if [[ -z "${QUASAR_CHECKPOINT_MANIFEST_URI:-}" ]]; then
  echo "error: set QUASAR_CHECKPOINT_MANIFEST_URI" >&2
  exit 2
fi
require_production_launch
echo "[orchestrator] run_id=${QUASAR_RUN_ID:-auto} netuid=${QUASAR_NETUID:-508}"
echo "[orchestrator] steps=${STEPS} timeout_sec=${TIMEOUT_SEC} poll=${POLL}"
echo "[orchestrator] checkpoint=${QUASAR_CHECKPOINT_MANIFEST_URI}"
echo "[orchestrator] auto_merge_release=${QUASAR_AUTO_MERGE_RELEASE} auto_release_checkpoints=${QUASAR_AUTO_RELEASE_CHECKPOINTS} merge_validators=${QUASAR_MERGE_VALIDATOR_HOTKEYS:-orchestrator-signer} validation_targets=${QUASAR_VALIDATION_TARGET_HOTKEYS:-${QUASAR_VALIDATOR_HOTKEYS:-merge-validators-only}} quorum=${QUASAR_MERGE_VERDICT_QUORUM:-majority}"
echo "[orchestrator] round_merge_grace min_sec=${QUASAR_ROUND_MERGE_MIN_GRACE_SEC} max_sec=${QUASAR_ROUND_MERGE_MAX_GRACE_SEC}"
echo "[orchestrator] require_registered_miners=${QUASAR_REQUIRE_REGISTERED_MINERS}"
echo "[orchestrator] gpu_policy min_to_join=${QUASAR_JOB_MIN_GPUS} job_gpu_count=${QUASAR_JOB_GPU_COUNT:-auto-detected-capacity}"
echo "[orchestrator] training_span local_steps=${QUASAR_LOCAL_STEPS} max_sequences=${QUASAR_MAX_SEQUENCES} seq_len=${QUASAR_SEQUENCE_LENGTH:-2048} batch=${QUASAR_BATCH_SIZE:-4}"
WANDB_PROJECT_EFFECTIVE="${QUASAR_ORCHESTRATOR_WANDB_PROJECT:-${QUASAR_WANDB_PROJECT:-${WANDB_PROJECT:-}}}"
if [[ -n "${WANDB_API_KEY:-}" || "${WANDB_MODE:-}" == "offline" ]]; then
  echo "[orchestrator] wandb=enabled project=${WANDB_PROJECT_EFFECTIVE:-quasar-incentive} mode=${WANDB_MODE:-online}"
else
  echo "[orchestrator] wandb=disabled set WANDB_API_KEY to enable"
fi
if [[ -n "${QUASAR_SHARD_MANIFEST_URIS:-${QUASAR_SHARD_MANIFEST_URI:-}}" ]]; then
  echo "[orchestrator] data_mode=explicit shards=${QUASAR_SHARD_MANIFEST_URIS:-${QUASAR_SHARD_MANIFEST_URI:-}}"
else
  echo "[orchestrator] data_mode=dynamic sources=${QUASAR_DATA_SOURCES:-all-approved} stages=${QUASAR_DATA_STAGES:-all}"
  echo "[orchestrator] auto_prepare=${QUASAR_AUTO_PREPARE_SHARDS} interval_sec=${QUASAR_AUTO_PREPARE_INTERVAL_SEC} tokens_per_shard=${QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD} min_shard_tokens=${QUASAR_MIN_SHARD_TOKENS}"
fi

args=(orchestrator run --steps "${STEPS}" --timeout-sec "${TIMEOUT_SEC}" --poll-interval "${POLL}")
if [[ -n "${QUASAR_RUN_ID:-}" ]]; then
  args+=(--run-id "${QUASAR_RUN_ID}")
fi
exec quasar-incentive "${args[@]}"
