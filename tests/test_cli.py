from pathlib import Path

from typer.testing import CliRunner

from hf_dataset_catalog.cli import app

runner = CliRunner()


def test_collect_help_lists_safe_scope() -> None:
    result = runner.invoke(app, ["collect", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--max-datasets" in result.stdout
    assert "--skip-papers" in result.stdout


def test_status_unknown_run_is_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["status", "--state-db", str(tmp_path / "x.duckdb"), "missing"],
    )
    assert result.exit_code != 0


def test_collect_rejects_nonpositive_smoke_limit() -> None:
    result = runner.invoke(app, ["collect", "--max-datasets", "0"])
    assert result.exit_code != 0
