from incentive.bucket import paths
from incentive.bucket.storage import LocalBucket
from incentive.dashboard.payload import build_dashboard_payload


def test_dashboard_counts_live_sync_accepted_updates(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-dashboard-live-sync"
    bucket.put_json(
        bucket.uri_for_key(f"{paths.live_fragment_merge_prefix(netuid, run_id, 7, 7)}/accepted_updates.json"),
        [
            {"hotkey": "miner-a", "weight": 1.0},
            {"hotkey": "miner-b", "weight": 2.0},
        ],
    )

    payload = build_dashboard_payload(bucket, netuid=netuid, run_id=run_id)

    assert payload["metrics"]["accepted_updates"] == 2
    assert payload["metrics"]["fragments_synced"] == 1
