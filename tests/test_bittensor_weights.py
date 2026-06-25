from incentive.chain.bittensor import BittensorAdapter
from incentive.config import ChainConfig


class _Hotkey:
    ss58_address = "validator-hotkey"


class _Wallet:
    hotkey = _Hotkey()


class _Metagraph:
    hotkeys = ["validator-hotkey", "registered-miner"]


class _Subtensor:
    def __init__(self):
        self.calls = []

    def metagraph(self, netuid: int):
        assert netuid == 24
        return _Metagraph()

    def set_weights(self, **kwargs):
        self.calls.append(kwargs)
        return (True, "Success")


def test_set_weights_drops_unregistered_hotkeys_before_chain_submit() -> None:
    subtensor = _Subtensor()
    adapter = BittensorAdapter(
        config=ChainConfig(netuid=24, wallet_name="wallet", hotkey_name="hotkey", dry_run=False),
        wallet=_Wallet(),
        subtensor=subtensor,
    )

    result = adapter.set_weights_from_scores(
        {
            "registered-miner": 1.0,
            "unregistered-miner": 100.0,
        },
        dry_run=False,
    )

    assert result["submitted"] is True
    assert result["uids"] == [1]
    assert result["weights"] == [1.0]
    assert result["dropped_hotkeys"] == ["unregistered-miner"]
    assert subtensor.calls[0]["uids"] == [1]
    assert subtensor.calls[0]["weights"] == [1.0]


def test_set_weights_does_not_submit_when_only_unregistered_hotkeys_have_scores() -> None:
    subtensor = _Subtensor()
    adapter = BittensorAdapter(
        config=ChainConfig(netuid=24, wallet_name="wallet", hotkey_name="hotkey", dry_run=False),
        wallet=_Wallet(),
        subtensor=subtensor,
    )

    result = adapter.set_weights_from_scores({"unregistered-miner": 1.0}, dry_run=False)

    assert result["submitted"] is False
    assert result["reason"] == "no_scores"
    assert result["dropped_hotkeys"] == ["unregistered-miner"]
    assert subtensor.calls == []
