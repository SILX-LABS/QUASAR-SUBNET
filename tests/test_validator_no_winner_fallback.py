from scripts.validator.service import (
    _resolve_no_winner_weight_target,
    _resolve_rescore_fallback_uid,
)


def test_no_winner_defaults_to_shared_burn_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_UID", raising=False)
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID", raising=False)

    uid, source = _resolve_no_winner_weight_target(
        subtensor=None,
        netuid=24,
        n_uids=256,
        king_uid=None,
        validator_uid=42,
        state_dir=str(tmp_path),
    )

    assert uid == 155
    assert source == "shared burn fallback"


def test_no_winner_keeps_incumbent_before_shared_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_UID", raising=False)

    uid, source = _resolve_no_winner_weight_target(
        subtensor=None,
        netuid=24,
        n_uids=256,
        king_uid=77,
        validator_uid=42,
        state_dir=str(tmp_path),
    )

    assert uid == 77
    assert source == "incumbent king"


def test_no_winner_can_override_shared_fallback_uid(monkeypatch, tmp_path):
    monkeypatch.setenv("QUASAR_NO_WINNER_FALLBACK_UID", "248")
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID", raising=False)

    uid, source = _resolve_no_winner_weight_target(
        subtensor=None,
        netuid=24,
        n_uids=256,
        king_uid=None,
        validator_uid=42,
        state_dir=str(tmp_path),
    )

    assert uid == 248
    assert source == "configured fallback"


def test_rescore_fallback_defaults_to_shared_burn_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("QUASAR_RESCORING_FALLBACK_UID", raising=False)
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_UID", raising=False)
    monkeypatch.delenv("QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID", raising=False)

    uid, source = _resolve_rescore_fallback_uid(
        subtensor=None,
        netuid=24,
        n_uids=256,
        validator_uid=42,
        state_dir=str(tmp_path),
    )

    assert uid == 155
    assert source == "shared burn fallback"
