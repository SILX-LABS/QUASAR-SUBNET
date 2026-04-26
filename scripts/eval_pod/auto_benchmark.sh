#!/bin/bash
# Auto-benchmark script for Subnet 24 king models (v3 — separate Lium pod)
# Triggered by validator when a new king is crowned
# Usage: auto_benchmark.sh <model_name> <king_uid>
#
# Spins up a fresh Lium pod, runs benchmarks, writes results, destroys pod.
# Does NOT interfere with eval pod.
set -e

MODEL="$1"
KING_UID="$2"
RESULTS_DIR="/root/benchmark_results"
LOG="/root/benchmark.log"
LOCK="/root/benchmark.lock"
BENCH_DIR="/home/affine-benchmark"
VENV="$BENCH_DIR/venv/bin"

# DynamoDB config for affine-benchmark results
# AWS creds live in /root/.aws_env (not checked into git). If missing, the
# DynamoDB write just fails non-fatally and the file-based summary still wins.
if [ -f /root/.aws_env ]; then
    # shellcheck disable=SC1091
    source /root/.aws_env
fi
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export DYNAMODB_TABLE="${DYNAMODB_TABLE:-distill_benchmark}"

if [ -z "$MODEL" ] || [ -z "$KING_UID" ]; then
    echo "Usage: auto_benchmark.sh <model_name> <king_uid>"
    exit 1
fi

# Check lock — don't run concurrent benchmarks
if [ -f "$LOCK" ]; then
    LOCK_PID=$(cat "$LOCK")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "Benchmark already running (PID $LOCK_PID). Skipping."
        exit 0
    else
        echo "Stale lock found. Removing."
        rm -f "$LOCK"
    fi
fi

# Create lock
echo $$ > "$LOCK"
cleanup() {
    # Kill SSH tunnel if we started it
    if [ -n "$TUNNEL_PID" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    rm -f "$LOCK"
}
trap cleanup EXIT

mkdir -p "$RESULTS_DIR"

echo "=== Auto-benchmark v3 (evalscope, tunnel to chat pod) for UID $KING_UID ($MODEL) ===" | tee "$LOG"
echo "Started at $(date -u)" | tee -a "$LOG"

# Use tunnel to chat pod (already has vLLM running with the king model)
CHAT_POD_HOST="91.224.44.207"
CHAT_POD_PORT="40070"
VLLM_PORT=8100

# Set up SSH tunnel to chat pod's vLLM
echo "Setting up tunnel to chat pod ($CHAT_POD_HOST:$CHAT_POD_PORT)..." | tee -a "$LOG"
ssh -f -N -L ${VLLM_PORT}:localhost:${VLLM_PORT} -p ${CHAT_POD_PORT} root@${CHAT_POD_HOST} -o StrictHostKeyChecking=no 2>/dev/null
TUNNEL_PID=$(pgrep -f "ssh.*-L ${VLLM_PORT}.*${CHAT_POD_HOST}" | head -1)
echo "Tunnel PID: $TUNNEL_PID" | tee -a "$LOG"

# Wait for vLLM to respond through tunnel
echo "Checking vLLM through tunnel..." | tee -a "$LOG"
SERVED=""
for i in $(seq 1 30); do
    SERVED=$(curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
    if [ -n "$SERVED" ]; then
        echo "vLLM serving: $SERVED (via tunnel)" | tee -a "$LOG"
        break
    fi
    sleep 5
done

if [ -z "$SERVED" ]; then
    echo "Cannot reach chat pod vLLM. Aborting." | tee -a "$LOG"
    exit 1
fi

# Set environment from .env
set -a
source "$BENCH_DIR/.env" 2>/dev/null || true
set +a
export USE_MODELSCOPE_HUB=True

# Run benchmarks via evalscope
cd "$BENCH_DIR"

BENCHMARKS="gsm8k humaneval ifeval bbh gpqa_diamond mmlu_pro math_500 aime25"
WORK_DIR="$RESULTS_DIR/evalscope_uid_${KING_UID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"

echo "Using fresh benchmark workdir: $WORK_DIR" | tee -a "$LOG"
echo "Running benchmarks: $BENCHMARKS" | tee -a "$LOG"

"$VENV/python" -c "
import os, sys, json, time
sys.path.insert(0, '$BENCH_DIR')
os.environ['USE_MODELSCOPE_HUB'] = 'True'

from runner import BenchmarkRunner

runner = BenchmarkRunner(
    api_url='http://localhost:$VLLM_PORT/v1',
    api_key='EMPTY',
    work_dir='$WORK_DIR',
    generation_config={
        'batch_size': 16,
        'do_sample': True,
        'temperature': 0.7,
        'max_tokens': 8192,
    },
)

benchmarks = '$BENCHMARKS'.split()
model = '$SERVED'  # Use the served model name (may differ from HF name)
uid = $KING_UID
results = {}

for i, bench in enumerate(benchmarks):
    print(f'[{i+1}/{len(benchmarks)}] Running {bench}...', flush=True)
    try:
        result = runner.run_benchmark(
            model_name=model,
            benchmark=bench,
            timeout=7200,
        )
        results[bench] = {
            'score': result.score,
            'metrics': result.metrics,
            'num_samples': result.num_samples,
            'eval_time_seconds': result.eval_time_seconds,
            'status': result.status,
            'error_message': result.error_message,
        }
        print(f'  {bench}: score={result.score:.4f} samples={result.num_samples} status={result.status}', flush=True)
    except Exception as e:
        print(f'  {bench}: FAILED — {e}', flush=True)
        results[bench] = {'score': 0.0, 'status': 'failed', 'error_message': str(e)}

# Write summary in format compatible with sync pipeline
summary = {
    'uid': uid,
    'model': '$MODEL',
    'completed': True,
    'tool': 'affine-benchmark/evalscope',
    'benchmarks': {b: r['score'] for b, r in results.items() if r.get('status') == 'succeeded'},
    'details': results,
}
summary_path = f'$RESULTS_DIR/uid_{uid}_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary written to {summary_path}')
print(json.dumps(summary, indent=2))

# Write to DynamoDB if credentials are set
try:
    import boto3
    from decimal import Decimal
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('distill_benchmark')
    revision = 'main'
    pk = f'$MODEL#{revision}'
    for bench, r in results.items():
        if r.get('status') == 'succeeded':
            table.put_item(Item={
                'PK': pk,
                'SK': bench,
                'model_name': '$MODEL',
                'revision': revision,
                'uid': uid,
                'score': Decimal(str(round(r['score'], 6))),
                'num_samples': r.get('num_samples', 0),
                'eval_time_seconds': Decimal(str(round(r.get('eval_time_seconds', 0), 1))),
                'status': 'completed',
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            print(f'  DynamoDB: wrote {bench} for {pk}', flush=True)
    print('DynamoDB writes complete', flush=True)
except Exception as e:
    print(f'DynamoDB write failed (non-fatal): {e}', flush=True)
" 2>&1 | tee -a "$LOG"

echo "=== All benchmarks complete at $(date -u) ===" | tee -a "$LOG"
