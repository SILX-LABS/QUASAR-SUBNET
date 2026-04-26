"""
Announcement queue and Discord posting for king changes.
"""
import json
import logging
import time

from eval.state import ValidatorState
from scripts.validator.config import DASHBOARD_URL, DISTIL_ROLE_ID, EVAL_PROMPTS_H2H, PAIRED_TEST_ALPHA, PUBLIC_API_URL

logger = logging.getLogger("quasar.validator")


def announce_new_king(new_uid, new_model, new_kl, old_uid, old_model, old_kl,
                      state: ValidatorState, paired_prompts=None, total_prompts=None,
                      p_value=None):
    """Write a pending announcement to state for async Discord posting."""
    kl_diff_pct = ((old_kl - new_kl) / old_kl * 100) if old_kl > 0 else 0

    # Fetch earnings data from prod API
    earnings_line = ""
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{PUBLIC_API_URL}/api/price", timeout=10)
        price_data = json.loads(resp.read())
        tao_per_day = price_data.get("miners_tao_per_day", 0)
        tao_usd = price_data.get("tao_usd", 0)
        usd_per_day = tao_per_day * tao_usd
        if tao_per_day > 0 and tao_usd > 0:
            earnings_line = (
                f"\n💰 **King earns ~{tao_per_day:.1f} τ/day (${usd_per_day:,.0f}/day)** — "
                f"winner takes all!\n"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch price data for announcement: {e}")

    role_ping = f"<@&{DISTIL_ROLE_ID}>"
    prompt_count = paired_prompts or total_prompts or EVAL_PROMPTS_H2H
    prompt_line = f"🧪 Compared on {prompt_count} paired prompts"
    if total_prompts and paired_prompts and total_prompts != paired_prompts:
        prompt_line = f"🧪 Compared on {paired_prompts}/{total_prompts} paired prompts"
    p_line = f" (p={p_value:.4f})" if isinstance(p_value, (int, float)) else ""
    announcement = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": False,
        "message": (
            f"{role_ping}\n"
            f"## 🏆 New King of Quasar SN24!\n\n"
            f"**UID {new_uid}** has dethroned **UID {old_uid}**\n\n"
            f"📊 **KL: {new_kl:.6f}** (previous king scored {old_kl:.6f} last eval)\n"
            f"{prompt_line}{p_line}\n"
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
            f"👑 Previous king: [{old_model}](<https://huggingface.co/{old_model}>)\n"
            f"{earnings_line}\n"
            f"Dethronement uses one-sided paired t-test (p<{PAIRED_TEST_ALPHA}). "
            f"Check the project docs on the dashboard for miner setup details.\n\n"
            f"📈 [Live Dashboard](<{DASHBOARD_URL}>)"
        ),
        "data": {
            "new_uid": new_uid, "new_model": new_model, "new_kl": new_kl,
            "old_uid": old_uid, "old_model": old_model, "old_kl": old_kl,
        },
    }
    state.save_announcement(announcement)
    logger.info(f"Announcement written: UID {new_uid} dethroned UID {old_uid}")
