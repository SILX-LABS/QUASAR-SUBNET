"""
Announcement queue for king changes.
"""
import json
import logging
import time

from eval.state import ValidatorState
from scripts.validator.config import DASHBOARD_URL, EVAL_PROMPTS_H2H, PUBLIC_API_URL, QUASAR_ROLE_ID
from scripts.validator.composite import (
    ARENA_V3_AXIS_WEIGHTS,
    BENCH_AXIS_WEIGHTS,
    COMPOSITE_SHADOW_VERSION,
)
from scripts.validator.single_eval import is_single_eval_mode

logger = logging.getLogger("quasar.validator")


_STRUCTURAL_AXES = (
    "on_policy_rkl",
    "kl",
    "capability",
    "judge_probe",
    "chat_turns_probe",
    "length",
    "degeneracy",
    "reasoning_density",
)


def _active_axis_summary() -> tuple[int, str]:
    active_bench = [k for k, w in BENCH_AXIS_WEIGHTS.items() if w > 0]
    active_arena = [k for k, w in ARENA_V3_AXIS_WEIGHTS.items() if w > 0]
    names = active_bench + active_arena + list(_STRUCTURAL_AXES)
    pretty = ", ".join(n.replace("_bench", "") for n in names)
    return len(names), pretty


def announce_new_king(new_uid, new_model, new_kl, old_uid, old_model, old_kl,
                      state: ValidatorState, paired_prompts=None, total_prompts=None,
                      p_value=None, *, new_composite_worst=None,
                      new_composite_weighted=None, new_limiting_axis=None,
                      old_composite_worst=None, old_composite_weighted=None):
    """Write a pending announcement to state for async posting."""
    single_eval_active = is_single_eval_mode()

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

    role_ping = f"<@&{QUASAR_ROLE_ID}>" if QUASAR_ROLE_ID else ""
    prompt_count = paired_prompts or total_prompts or EVAL_PROMPTS_H2H
    prompt_line = f"🧪 Scored on {prompt_count} block-seeded prompts"
    n_axes, axis_list = _active_axis_summary()
    gate_explainer = (
        f"Dethronement uses an absolute composite across {n_axes} weighted axes "
        f"({axis_list}). Clear win on `worst` (>3% margin) takes the crown; "
        f"the tied region falls back to `weighted` with the same margin. "
        f"KL is a distillation axis, not the king-selection gate. "
        f"(composite schema v{COMPOSITE_SHADOW_VERSION})"
    )
    p_line = f" (p={p_value:.4f})" if isinstance(p_value, (int, float)) else ""

    if new_composite_worst is not None:
        worst_line = f"📊 **Composite worst: {new_composite_worst:.3f}**"
        if new_limiting_axis:
            axis_pretty = new_limiting_axis.replace("_bench", "").replace("_", " ")
            worst_line += f" (limiting axis: {axis_pretty})"
        if old_composite_worst is not None:
            worst_line += f" — previous king: {old_composite_worst:.3f}"
        weighted_part = ""
        if new_composite_weighted is not None:
            weighted_part = f"\n📐 Weighted mean: {new_composite_weighted:.3f}"
            if old_composite_weighted is not None:
                weighted_part += f" (was {old_composite_weighted:.3f})"
        headline = (
            worst_line
            + weighted_part
            + f"\n└ KL component: {new_kl:.6f} (one scored axis, not the ranking key)"
        )
    else:
        headline = f"📊 **KL: {new_kl:.6f}** (one scored axis; previous king KL: {old_kl:.6f})"

    announcement = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": False,
        "message": (
            f"{role_ping}\n"
            f"## 🏆 New King of Quasar SN24!\n\n"
            f"**UID {new_uid}** has dethroned **UID {old_uid}**\n\n"
            f"{headline}\n"
            f"{prompt_line}{p_line}\n"
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
            f"👑 Previous king: [{old_model}](<https://huggingface.co/{old_model}>)\n"
            f"{earnings_line}\n"
            f"{gate_explainer} "
            f"Check the mining guide for setup details.\n\n"
            f"📈 [Live Dashboard](<{DASHBOARD_URL}>)"
        ),
        "data": {
            "new_uid": new_uid, "new_model": new_model, "new_kl": new_kl,
            "old_uid": old_uid, "old_model": old_model, "old_kl": old_kl,
            "new_composite_worst": new_composite_worst,
            "new_composite_weighted": new_composite_weighted,
            "new_limiting_axis": new_limiting_axis,
            "old_composite_worst": old_composite_worst,
            "old_composite_weighted": old_composite_weighted,
            "single_eval_mode": single_eval_active,
        },
    }
    state.save_announcement(announcement)
    logger.info(f"Announcement written: UID {new_uid} dethroned UID {old_uid}")
