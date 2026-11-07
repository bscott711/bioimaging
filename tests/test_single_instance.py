"""Only one opym-backfill orchestrator may run on a registry (review CL-09).

A manual one-shot or ``--retry-failed`` while the service is mid-pass would
run a second orchestrator on the same registry, pool and tickets.
"""

import errno

import pytest

import run_backfill_cli as cli


def test_second_lock_on_same_registry_is_refused(tmp_path):
    registry = tmp_path / "registry.sqlite3"
    first = cli.acquire_single_instance_lock(registry)
    try:
        assert (tmp_path / cli.LOCK_NAME).is_file()
        with pytest.raises(cli.AlreadyRunningError, match="already running"):
            cli.acquire_single_instance_lock(registry)
    finally:
        first.close()


def test_lock_is_free_again_after_release(tmp_path):
    registry = tmp_path / "registry.sqlite3"
    cli.acquire_single_instance_lock(registry).close()
    cli.acquire_single_instance_lock(registry).close()


def test_locks_on_different_registries_do_not_collide(tmp_path):
    a = cli.acquire_single_instance_lock(tmp_path / "a" / "registry.sqlite3")
    b = cli.acquire_single_instance_lock(tmp_path / "b" / "registry.sqlite3")
    a.close()
    b.close()


def _run_main(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(cli, "run_backfill", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(cli, "watch_backfill", lambda *a, **k: calls.append("watch"))
    monkeypatch.setattr(cli.sys, "argv", ["opym-backfill", *argv])
    return calls


def test_main_exits_nonzero_and_does_nothing_while_lock_held(monkeypatch, tmp_path, capsys):
    registry = tmp_path / "registry.sqlite3"
    holder = cli.acquire_single_instance_lock(registry)
    try:
        calls = _run_main(
            monkeypatch,
            ["--registry-path", str(registry), "--roots", str(tmp_path), "--retry-failed"],
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert calls == []                      # no pass started
        assert not registry.exists()            # --retry-failed never touched the registry
        assert "already running" in capsys.readouterr().err
    finally:
        holder.close()


def test_main_dry_run_is_allowed_while_lock_held(monkeypatch, tmp_path):
    registry = tmp_path / "registry.sqlite3"
    holder = cli.acquire_single_instance_lock(registry)
    try:
        calls = _run_main(
            monkeypatch,
            ["--registry-path", str(registry), "--roots", str(tmp_path), "--dry-run"],
        )
        cli.main()
        assert calls == ["run"]
    finally:
        holder.close()


def test_main_runs_when_lock_is_free(monkeypatch, tmp_path):
    calls = _run_main(
        monkeypatch,
        ["--registry-path", str(tmp_path / "registry.sqlite3"), "--roots", str(tmp_path)],
    )
    cli.main()
    assert calls == ["run"]


def test_unavailable_lock_fails_open_with_warning(monkeypatch, tmp_path, capsys):
    """ENOLCK is not "already running": a restart loop would leave the backfill down."""
    def broken_flock(*a, **k):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(cli.fcntl, "flock", broken_flock)
    calls = _run_main(
        monkeypatch,
        ["--registry-path", str(tmp_path / "registry.sqlite3"), "--roots", str(tmp_path)],
    )
    cli.main()
    assert calls == ["run"]
    err = capsys.readouterr().err
    assert "WARNING: single-instance lock unavailable (ENOLCK)" in err
    assert "WITHOUT the guard" in err


def test_unwritable_lock_dir_fails_open_with_warning(monkeypatch, tmp_path, capsys):
    def broken_open(*a, **k):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr("builtins.open", broken_open)
    assert cli.acquire_single_instance_lock(tmp_path / "registry.sqlite3") is None
    assert "(EIO)" in capsys.readouterr().err
