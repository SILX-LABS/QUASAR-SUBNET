import logging

from scripts.validator.chat_pod import ensure_chat_server_running, restart_chat_server, trigger_benchmarks

logger = logging.getLogger("quasar.validator")


def sync_king_runtime(king_changed, king_model, king_uid):
    if not king_model:
        return
    if king_changed:
        restart_chat_server(king_model)
        trigger_benchmarks(king_model, king_uid)
        return
    ensure_chat_server_running(king_model)
