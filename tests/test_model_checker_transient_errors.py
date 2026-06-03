from eval.model_checker import _is_transient_error, _repo_access_failure_reason


def test_network_unreachable_is_retryable_not_hard_dq():
    exc = RuntimeError("[Errno 101] Network is unreachable")

    assert _is_transient_error(exc)
    assert _repo_access_failure_reason("miner/model", exc) is None


def test_dns_resolution_failure_is_retryable_not_hard_dq():
    exc = RuntimeError("[Errno -3] Temporary failure in name resolution")

    assert _is_transient_error(exc)
    assert _repo_access_failure_reason("miner/model", exc) is None


def test_private_or_missing_repo_remains_hard_dq():
    exc = RuntimeError("404 Client Error: Repository Not Found")

    assert _repo_access_failure_reason("miner/model", exc) is not None
