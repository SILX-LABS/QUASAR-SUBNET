import json
from types import SimpleNamespace

from eval.chain import parse_commitments
from scripts.validator.challengers import select_challengers
from scripts.validator.coordination import (
    CoordinationRound,
    configured_start_block,
    parse_commitments_at_cutoff,
    reset_anchor_challenger_uids,
    round_start_wait_target,
    scheduled_challenger_uids,
)
from scripts.validator.service import (
    _capture_coordination_chain_snapshot,
    _resolve_coordinated_king,
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


def _model(uid, owner, block, hotkey=None):
    return {
        "model": f"{owner}/model-{uid}",
        "revision": "abc",
        "commit_block": block,
        "hotkey": hotkey or f"hk-{uid}",
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


def test_configured_start_block_waits_until_requested_block(monkeypatch):
    monkeypatch.setenv("QUASAR_COORDINATION_START_BLOCK", "8381520")

    assert configured_start_block() == 8381520
    assert round_start_wait_target(8378786) == (8381520, "configured_start")


def test_configured_start_block_allows_requested_round_window(monkeypatch):
    monkeypatch.setenv("QUASAR_COORDINATION_START_BLOCK", "8381520")

    assert round_start_wait_target(8381520) == (None, "round_start")


def test_default_configured_start_block_targets_clean_restart(monkeypatch):
    monkeypatch.delenv("QUASAR_COORDINATION_START_BLOCK", raising=False)
    monkeypatch.delenv("DISTIL_COORDINATION_START_BLOCK", raising=False)

    assert configured_start_block() == 8382960
    assert round_start_wait_target(8381520) == (8382960, "configured_start")


def test_invalid_configured_start_block_is_ignored(monkeypatch):
    monkeypatch.setenv("QUASAR_COORDINATION_START_BLOCK", "not-a-block")

    assert configured_start_block() is None
    assert round_start_wait_target(8378786) == (None, "round_start")


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


class _SnapshotSubtensor:
    def __init__(self):
        self.permit_calls = 0
        self.stake_calls = 0
        self.weight_calls = 0

    def get_subnet_validator_permits(self, netuid, block=None):
        self.permit_calls += 1
        assert block == 8382960
        return [True] * 256

    def get_stake_weight(self, netuid, block=None):
        self.stake_calls += 1
        assert block == 8382960
        return [1.0] * 256

    def weights(self, netuid, block=None):
        self.weight_calls += 1
        assert block == 8382960
        return [(0, [(155, 65535)])]


class _ExplodingSubtensor:
    def get_subnet_validator_permits(self, *args, **kwargs):
        raise AssertionError("captured snapshot should avoid permit refetch")

    def get_stake_weight(self, *args, **kwargs):
        raise AssertionError("captured snapshot should avoid stake refetch")

    def weights(self, *args, **kwargs):
        raise AssertionError("captured snapshot should avoid weight refetch")


def test_coordinated_king_uses_captured_chain_snapshot(tmp_path):
    coord_round = _round(11643)
    subtensor = _SnapshotSubtensor()

    snapshot = _capture_coordination_chain_snapshot(
        subtensor, None, 24, 256, coord_round, str(tmp_path),
    )

    assert snapshot["weight_block"] == 8382960
    assert snapshot["chain_king_uid"] == 155
    assert subtensor.permit_calls == 1
    assert subtensor.stake_calls == 1
    assert subtensor.weight_calls == 1

    state = SimpleNamespace(h2h_latest={}, scores={"155": 2.5}, composite_scores={})
    king_uid, king_kl, source = _resolve_coordinated_king(
        _ExplodingSubtensor(),
        None,
        24,
        256,
        {155: _model(155, "silx", 8380400)},
        state,
        str(tmp_path),
        coord_round=coord_round,
        chain_snapshot=snapshot,
    )

    assert (king_uid, king_kl, source) == (155, 2.5, "chain_consensus")


def test_scheduled_challengers_rate_limits_hf_owner_window():
    models = {
        11: _model(11, "Olague-Secret", 100),
        12: _model(12, "Olague-Secret", 200),
        13: _model(13, "Olague-Secret", 300),
        14: _model(14, "Atom0124", 400),
        15: _model(15, "Atom0124", 500),
        16: _model(16, "Atom0124", 600),
        17: _model(17, "solo", 700),
    }

    assert scheduled_challenger_uids(
        models, _round(), 5, pending_uids=set(models),
    ) == [11, 12, 14, 15, 17]


def test_scheduled_challengers_rate_limit_counts_already_scored_models():
    models = {
        11: _model(11, "Olague-Secret", 100),
        12: _model(12, "Olague-Secret", 200),
        13: _model(13, "Olague-Secret", 300),
        14: _model(14, "Olague-Secret", 400),
        20: _model(20, "Atom0124", 500),
    }

    assert scheduled_challenger_uids(
        models, _round(), 5, pending_uids={13, 14, 20},
    ) == [20]


def test_scheduled_challengers_rate_limits_hotkey_window():
    models = {
        11: _model(11, "alpha", 100, hotkey="hk-shared"),
        12: _model(12, "beta", 200, hotkey="hk-shared"),
        13: _model(13, "gamma", 300, hotkey="hk-shared"),
        14: _model(14, "delta", 400, hotkey="hk-free"),
    }

    assert scheduled_challenger_uids(
        models, _round(), 5, pending_uids=set(models),
    ) == [11, 12, 14]


def test_reset_anchor_respects_round_cap():
    models = _models([
        (21, 2100),
        (22, 2200),
        (23, 2300),
        (24, 2400),
        (25, 2500),
    ])

    assert reset_anchor_challenger_uids(models, 5) == [21, 22, 23, 24, 25]


class _State:
    def __init__(self, state_dir):
        self.state_dir = state_dir
        self.evaluated_uids = set()
        self.composite_scores = {}
        self.scores = {}
        self.h2h_latest = {"king_uid": 169}
        self.h2h_history = []
        self.model_hashes = {}
        self.permanently_bad_models = set()
        self.model_score_history = {}
        self.eval_progress = {}


def test_single_eval_skips_recycled_uid_with_scored_content_hash(tmp_path):
    state = _State(tmp_path)
    state.composite_scores = {
        "11": {
            "model": "old/scored",
            "revision": "rev-old",
            "block": 100,
            "weighted": 0.42,
            "worst": 0.1,
        }
    }
    (tmp_path / "model_content_hashes.json").write_text(json.dumps({
        "11": "same-content",
        "20": "same-content",
        "21": "new-content",
    }))
    valid_models = {
        20: _model(20, "copy", 200),
        21: _model(21, "fresh", 300),
    }

    challengers = select_challengers(
        valid_models, state, king_uid=169, king_kl=3.0,
        epoch_count=1, coord_round=_round(),
    )

    assert list(challengers) == [21]


def test_single_eval_does_not_maintenance_schedule_already_scored_copy(tmp_path):
    state = _State(tmp_path)
    state.composite_scores = {
        "11": {
            "model": "old/scored",
            "revision": "rev-old",
            "block": 100,
            "weighted": 0.42,
            "worst": 0.1,
        }
    }
    (tmp_path / "model_content_hashes.json").write_text(json.dumps({
        "11": "same-content",
        "20": "same-content",
    }))
    valid_models = {20: _model(20, "copy", 200)}

    challengers = select_challengers(
        valid_models, state, king_uid=169, king_kl=3.0,
        epoch_count=1, coord_round=_round(),
    )

    assert challengers == {}


def test_single_eval_skips_recently_scored_same_model_repo(tmp_path):
    state = _State(tmp_path)
    state.h2h_history = [{
        "block": _round().round_start_block - 10,
        "results": [{
            "uid": 11,
            "model": "owner/repo",
            "kl": 2.5,
        }],
    }]
    valid_models = {
        20: {
            "model": "owner/repo",
            "revision": "new-revision",
            "commit_block": 200,
            "hotkey": "hk-20",
        },
        21: _model(21, "fresh", 300),
    }

    challengers = select_challengers(
        valid_models, state, king_uid=169, king_kl=3.0,
        epoch_count=1, coord_round=_round(),
    )

    assert list(challengers) == [21]


class _Metagraph:
    hotkeys = ["hk0", "hk1"]
    coldkeys = ["ck0", "ck1"]


def test_parse_commitments_uses_chain_block_over_payload_block():
    revealed = {
        "hk1": [
            (
                8342504,
                json.dumps({
                    "model": "william-777/creation-4",
                    "revision": "903bb364ed04",
                    "block": 8330999,
                    "hotkey": "payload-hotkey",
                }),
            )
        ]
    }

    commitments, uid_to_hotkey, _ = parse_commitments(_Metagraph(), revealed, 2)

    assert uid_to_hotkey[1] == "hk1"
    assert commitments[1]["block"] == 8342504
    assert commitments[1]["hotkey"] == "hk1"


def test_parse_commitments_at_cutoff_uses_chain_block_over_payload_block():
    revealed = {
        "hk1": [
            (
                8342504,
                json.dumps({
                    "model": "william-777/creation-4",
                    "revision": "903bb364ed04",
                    "block": 8330999,
                    "hotkey": "payload-hotkey",
                }),
            )
        ]
    }

    commitments, uid_to_hotkey, _ = parse_commitments_at_cutoff(
        _Metagraph(), revealed, 2, cutoff_block=8342620,
    )

    assert uid_to_hotkey[1] == "hk1"
    assert commitments[1]["block"] == 8342504
    assert commitments[1]["hotkey"] == "hk1"
