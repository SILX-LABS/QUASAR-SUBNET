import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.runtime import (
    ALLOWED_ORIGINS,
    CACHE_TTL,
    CHAT_POD_APP_PORT,
    CHAT_POD_HOST,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    DASHBOARD_URL,
    DISK_CACHE_DIR,
    NETUID,
    STATE_DIR,
    TMC_BASE,
    TMC_HEADERS,
)

CHAT_POD_PORT = CHAT_POD_APP_PORT

# ── Tunables (single source of truth, duplicated in several places before) ──
STALE_EVAL_BLOCKS = 50
EPOCH_BLOCKS = 360
CHAT_RESTART_COOLDOWN = 120
CHAT_SERVER_SCRIPT = "/root/chat_server.py"
MAX_COMPARE_UIDS = 10
MAX_BATCH_UIDS = 64
ANNOUNCEMENT_CLAIMS_KEEP = 50

API_DESCRIPTION = f"""
# Quasar - Subnet {NETUID} API

Public API for [Quasar]({DASHBOARD_URL}), a Bittensor subnet where miners compete to produce the best knowledge-distilled small language models.

## How It Works

Miners submit Quasar-compatible models and validators score them with the composite evaluator. KL is one distillation axis, while king selection uses composite worst/weighted scores.

## Quick Start

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/scores
curl http://localhost:8000/api/price
```
"""
