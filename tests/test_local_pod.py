import shlex
import sys

from eval.local_pod import LocalPodManager


def test_local_pod_exec_upload_download(tmp_path):
    backend = LocalPodManager(work_dir=tmp_path / "runs", python_bin=sys.executable)
    backend.connect()

    source = tmp_path / "source.txt"
    source.write_text("hello local eval")
    remote = tmp_path / "runs" / "run1" / "remote.txt"

    backend.upload(str(source), str(remote))
    assert remote.read_text() == "hello local eval"

    result = backend.exec(f"{shlex.quote(sys.executable)} -c 'print(123)'", timeout=10)
    assert result["success"]
    assert result["stdout"].strip() == "123"

    downloaded = tmp_path / "downloaded.txt"
    backend.download(str(remote), str(downloaded))
    assert downloaded.read_text() == "hello local eval"

    run_dir = tmp_path / "runs" / "distil_eval_1"
    run_dir.mkdir()
    (run_dir / "eval_results.json").write_text("{}")
    backend.register_run_dir(str(run_dir))
    backend.post_eval_cleanup("teacher/model")
    assert not run_dir.exists()
