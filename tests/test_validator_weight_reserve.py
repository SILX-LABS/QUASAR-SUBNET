import pytest

from eval.chain import build_winner_take_all_weights


@pytest.fixture(autouse=True)
def clear_weight_reserve_env(monkeypatch):
    for name in (
        "QUASAR_WEIGHT_RESERVE_FRACTION",
        "QUASAR_WEIGHT_RESERVE_UID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_weights_leave_ten_percent_reserve_on_uid155():
    weights = build_winner_take_all_weights(256, 167)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[167] == pytest.approx(0.90)
    assert weights[155] == pytest.approx(0.10)


def test_weight_reserve_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QUASAR_WEIGHT_RESERVE_FRACTION", "0")

    weights = build_winner_take_all_weights(256, 167)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[167] == pytest.approx(1.0)
    assert weights[155] == pytest.approx(0.0)


def test_weight_reserve_uid_can_be_overridden(monkeypatch):
    monkeypatch.setenv("QUASAR_WEIGHT_RESERVE_FRACTION", "0.10")
    monkeypatch.setenv("QUASAR_WEIGHT_RESERVE_UID", "248")

    weights = build_winner_take_all_weights(256, 167)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[167] == pytest.approx(0.90)
    assert weights[248] == pytest.approx(0.10)
    assert weights[155] == pytest.approx(0.0)


def test_weight_reserve_ignored_when_reserve_is_winner(monkeypatch):
    monkeypatch.setenv("QUASAR_WEIGHT_RESERVE_FRACTION", "0.10")
    monkeypatch.setenv("QUASAR_WEIGHT_RESERVE_UID", "167")

    weights = build_winner_take_all_weights(256, 167)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[167] == pytest.approx(1.0)
    assert weights[155] == pytest.approx(0.0)
