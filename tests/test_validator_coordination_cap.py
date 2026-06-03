from scripts.validator.coordination import (
    CoordinationRound,
    reset_anchor_challenger_uids,
    scheduled_challenger_uids,
)


def _round(round_id: int = 100) -> CoordinationRound:
    return CoordinationRound(
        protocol_version=2,
        round_id=round_id,
        round_blocks=720,
        round_start_block=round_id * 720,
        commit_cutoff_block=round_id * 720 - 20,
        eval_seed_block=round_id * 720,
        activation_block=round_id * 720 + 600,
        commit_finality_blocks=20,
        activation_delay_blocks=600,
    )


def _models(blocks):
    return {
        uid: {
            "model": f"miner/{uid}",
            "revision": "abc",
            "commit_block": block,
        }
        for uid, block in blocks
    }


def test_scheduled_challengers_cap_to_five_oldest_pending():
    models = _models([
        (11, 100),
        (12, 200),
        (13, 300),
        (14, 400),
        (15, 500),
        (16, 600),
    ])

    assert scheduled_challenger_uids(models, _round(), 5) == [11, 12, 13, 14, 15]


def test_scheduled_challengers_next_round_takes_remaining_oldest():
    # Caller filters out models already scored locally. The scheduler should
    # then keep FIFO order across the remaining backlog.
    models = _models([
        (15, 500),
        (16, 600),
        (17, 700),
        (18, 800),
        (19, 900),
    ])

    assert scheduled_challenger_uids(models, _round(), 5) == [15, 16, 17, 18, 19]


def test_reset_anchor_respects_round_cap():
    models = _models([
        (21, 2100),
        (22, 2200),
        (23, 2300),
        (24, 2400),
        (25, 2500),
    ])

    assert reset_anchor_challenger_uids(models, 5) == [21, 22, 23, 24, 25]
