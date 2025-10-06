from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml

from timegpt_v2.fe.base_features import build_feature_matrix
from timegpt_v2.fe.context import FeatureContext
from timegpt_v2.io.gcs_reader import GCSReader, ReaderConfig
from timegpt_v2.quality.checks import DataQualityChecker
from timegpt_v2.quality.contracts import DataQualityPolicy

app = typer.Typer(help="TimeGPT Intraday v2 command line interface")

CONFIG_DIR_OPTION = typer.Option(
    Path("configs"),
    "--config-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Directory containing configuration files",
)
RUN_ID_OPTION = typer.Option(..., "--run-id", help="Unique run identifier")
GRID_CONFIG_OPTION = typer.Option(
    None,
    "--grid-config",
    exists=True,
    file_okay=False,
    dir_okay=False,
    help="Optional trading grid override",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if isinstance(loaded, MutableMapping):
        return dict(loaded)
    raise typer.BadParameter(f"{path} must contain a mapping")


def _expect_mapping(name: str, payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise typer.BadParameter(f"{name} must be a mapping")
    return payload


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()  # type: ignore[return-value]
    return str(obj)


@app.command(name="check-data")
def check_data(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Validate source data before downstream processing."""
    started = datetime.utcnow()
    run_dir = Path("artifacts") / "runs" / run_id
    validation_dir = run_dir / "validation"
    logs_dir = run_dir / "logs"
    validation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "loader.log"

    universe_cfg = _load_yaml(config_dir / "universe.yaml")
    data_cfg = _load_yaml(config_dir / "data.yaml")
    policy_cfg = _load_yaml(config_dir / "dq_policy.yaml")

    tickers_raw = universe_cfg.get("tickers", [])
    if not isinstance(tickers_raw, Sequence):
        raise typer.BadParameter("universe.tickers must be a sequence")
    tickers = [str(t) for t in tickers_raw]

    date_cfg = _expect_mapping("universe.dates", universe_cfg.get("dates"))
    try:
        start_str = str(date_cfg["start"])
        end_str = str(date_cfg["end"])
    except KeyError as exc:
        raise typer.BadParameter(f"Missing universe date config: {exc}") from exc
    start_date = datetime.fromisoformat(start_str + "T00:00:00").date()
    end_date = datetime.fromisoformat(end_str + "T00:00:00").date()

    gcs_cfg = _expect_mapping("data.gcs", data_cfg.get("gcs"))
    bucket = str(gcs_cfg.get("bucket", ""))
    template = str(gcs_cfg.get("template", ""))
    if not template:
        raise typer.BadParameter("configs/data.yaml must define gcs.template")

    reader = GCSReader(ReaderConfig(bucket=bucket, template=template))

    raw_frame = reader.read_universe(tickers, start_date, end_date)
    checker = DataQualityChecker(policy=DataQualityPolicy.from_dict(policy_cfg))
    clean_frame, report = checker.validate(raw_frame)

    report_path = validation_dir / "dq_report.json"
    clean_path = validation_dir / "clean.parquet"
    clean_frame.to_parquet(clean_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, default=_json_default)

    meta_path = run_dir / "meta.json"
    finished = datetime.utcnow()
    meta_payload = {
        "command": "check-data",
        "run_id": run_id,
        "config_dir": str(config_dir),
        "started_at": started.isoformat() + "Z",
        "finished_at": finished.isoformat() + "Z",
        "status": "passed" if report.passed else "failed",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)

    _append_log(
        log_path,
        {
            "ts": finished.isoformat() + "Z",
            "event": "check_data",
            "rows_before": report.rows_before,
            "rows_after": report.rows_after,
            "passed": report.passed,
        },
    )

    if not report.passed:
        raise typer.Exit(code=1)


@app.command(name="build-features")
def build_features(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Build feature matrix from validated data."""
    run_dir = Path("artifacts") / "runs" / run_id
    validation_dir = run_dir / "validation"
    output_dir = run_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = validation_dir / "clean.parquet"
    if not clean_path.exists():
        raise typer.BadParameter(
            "Validated data not found. Run `check-data` before `build-features`."
        )

    data = pd.read_parquet(clean_path)
    context = FeatureContext(symbols=data["symbol"].unique())
    features = build_feature_matrix(data, context=context)

    feature_path = output_dir / "features.parquet"
    features.to_parquet(feature_path, index=False)

    meta_path = run_dir / "meta.json"
    meta = {
        "command": "build-features",
        "run_id": run_id,
        "rows": len(features),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_path": str(clean_path),
        "output_path": str(feature_path),
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.setdefault("steps", {})
    existing["command"] = "build-features"
    existing["steps"]["build-features"] = meta
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    typer.echo(f"Features written to {feature_path}")


@app.command()
def forecast(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Generate TimeGPT quantile forecasts."""
    typer.echo(f"forecast stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def backtest(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Run trading backtest using generated forecasts."""
    typer.echo(f"backtest stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def evaluate(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Evaluate model and trading performance metrics."""
    typer.echo(f"evaluate stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def report(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Assemble final report for the run."""
    typer.echo(f"report stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def sweep(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
    grid_config: Path | None = GRID_CONFIG_OPTION,
) -> None:
    """Execute trading parameter sweep."""
    typer.echo(f"sweep stub: run_id={run_id}, config_dir={config_dir}, grid={grid_config}")


@app.command()
def run(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Convenience pipeline runner (scaffold placeholder)."""
    typer.echo(f"run stub: run_id={run_id}, config_dir={config_dir}")


def _append_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
