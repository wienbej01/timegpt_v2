from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import typer
import yaml

from timegpt_v2.backtest.grid import GridSearch
from timegpt_v2.fe.base_features import build_feature_matrix
from timegpt_v2.fe.context import FeatureContext
from timegpt_v2.forecast.scheduler import ForecastScheduler, get_trading_holidays
from timegpt_v2.forecast.timegpt_client import TimeGPTClient, TimeGPTConfig
from timegpt_v2.framing.build_payloads import build_x_df_for_horizon, build_y_df
from timegpt_v2.io.gcs_reader import GCSReader, ReaderConfig
from timegpt_v2.quality.checks import DataQualityChecker
from timegpt_v2.quality.contracts import DataQualityPolicy
from timegpt_v2.utils.cache import ForecastCache

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


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("timegpt_v2.forecast")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


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

    run_dir = Path("artifacts") / "runs" / run_id
    feature_path = run_dir / "features" / "features.parquet"
    forecasts_dir = run_dir / "forecasts"
    logs_dir = run_dir / "logs"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not feature_path.exists():
        raise typer.BadParameter("Feature matrix not found. Run `build-features` first.")

    features = pd.read_parquet(feature_path)
    if "timestamp" not in features.columns or "symbol" not in features.columns:
        raise typer.BadParameter("features.parquet must include 'timestamp' and 'symbol' columns")

    forecast_cfg = _load_yaml(config_dir / "forecast.yaml")
    quantiles = tuple(float(q) for q in forecast_cfg.get("quantiles", [0.25, 0.5, 0.75]))
    horizon = int(forecast_cfg.get("horizon_min", 15))
    freq = str(forecast_cfg.get("freq", "min"))
    tz_name = str(forecast_cfg.get("tz", "America/New_York"))
    snapshot_strs = forecast_cfg.get("snapshots_et", [])
    if not snapshot_strs:
        raise typer.BadParameter("forecast.yaml must define snapshots_et")

    zone = ZoneInfo(tz_name)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features_et = features["timestamp"].dt.tz_convert(zone)
    features = features.assign(timestamp_et=features_et)

    snapshot_times: list[time] = []
    for snapshot in snapshot_strs:
        try:
            parsed = datetime.strptime(str(snapshot), "%H:%M").time()
        except ValueError as exc:  # pragma: no cover - config error
            raise typer.BadParameter(f"Invalid snapshot time: {snapshot}") from exc
        snapshot_times.append(parsed)

    trade_dates = sorted(features["timestamp_et"].dt.date.unique())
    holidays = get_trading_holidays(years=sorted(list(set(d.year for d in trade_dates))))
    scheduler = ForecastScheduler(
        dates=trade_dates, snapshots=snapshot_times, tz=zone, holidays=holidays
    )
    snapshots = scheduler.generate_snapshots()

    logger = _configure_logger(logs_dir / "forecast.log")
    cache = ForecastCache(forecasts_dir / "cache", logger=logger)
    client = TimeGPTClient(
        cache=cache,
        config=TimeGPTConfig(freq=freq, horizon=horizon, quantiles=quantiles),
        logger=logger,
    )

    expected_symbols = sorted(str(sym) for sym in features["symbol"].unique())
    if not expected_symbols:
        raise typer.BadParameter("No symbols available for forecasting")

    results: list[pd.DataFrame] = []

    for snapshot_local in snapshots:
        snapshot_utc = pd.Timestamp(snapshot_local).tz_convert("UTC")
        history = build_y_df(features, snapshot_utc)
        if history.empty:
            continue
        latest = history.groupby("unique_id")["ds"].max()
        if not (latest == snapshot_utc).all():
            continue
        future = build_x_df_for_horizon(features, snapshot_utc, horizon)
        forecast_df = client.forecast(
            history,
            future,
            snapshot_ts=snapshot_utc,
            horizon=horizon,
            freq=freq,
            quantiles=quantiles,
        )
        if forecast_df.empty:
            raise typer.Exit(code=1)
        forecast_df["snapshot_utc"] = snapshot_utc
        results.append(forecast_df)

    if not results:
        raise typer.Exit(code=1)

    combined = pd.concat(results, ignore_index=True)
    quantile_cols = [col for col in combined.columns if col.startswith("q")]

    for _snapshot_ts, group in combined.groupby("snapshot_utc"):
        group_symbols = set(group["unique_id"].astype(str))
        if group_symbols != set(expected_symbols):
            raise typer.Exit(code=1)
        for column in quantile_cols:
            if group[column].isna().any():
                raise typer.Exit(code=1)

    snapshot_values = sorted(ts.isoformat() for ts in combined["snapshot_utc"].unique())

    output = combined.rename(columns={"unique_id": "symbol", "forecast_ts": "ts_utc"})
    output["ts_utc"] = pd.to_datetime(output["ts_utc"], utc=True)
    output.sort_values(["ts_utc", "symbol"], inplace=True)
    ts_strings = output["ts_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output.loc[:, "ts_utc"] = ts_strings.to_numpy(dtype=object)
    output = output[["ts_utc", "symbol", *quantile_cols]]
    output_path = forecasts_dir / "quantiles.csv"
    output.to_csv(output_path, index=False)

    meta_path = run_dir / "meta.json"
    meta_entry = {
        "command": "forecast",
        "run_id": run_id,
        "rows": len(output),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_path": str(output_path),
        "snapshots": snapshot_values,
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.setdefault("steps", {})
    existing["command"] = "forecast"
    existing["steps"]["forecast"] = meta_entry
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    typer.echo(f"Forecast quantiles written to {output_path}")


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
    run_dir = Path("artifacts") / "runs" / run_id
    forecasts_path = run_dir / "forecasts" / "quantiles.csv"
    features_path = run_dir / "features" / "features.parquet"
    prices_path = run_dir / "validation" / "clean.parquet"  # Use clean data as prices
    output_dir = run_dir / "eval" / "grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not forecasts_path.exists():
        raise typer.BadParameter("Forecasts not found. Run `forecast` first.")
    if not features_path.exists():
        raise typer.BadParameter("Features not found. Run `build-features` first.")
    if not prices_path.exists():
        raise typer.BadParameter("Prices not found. Run `check-data` first.")

    forecasts = pd.read_csv(forecasts_path)
    features = pd.read_parquet(features_path)
    prices = pd.read_parquet(prices_path)

    trading_cfg_path = grid_config or config_dir / "trading.yaml"
    trading_cfg = _load_yaml(trading_cfg_path)

    logger = _configure_logger(run_dir / "logs" / "sweep.log")
    grid_search = GridSearch(trading_cfg=trading_cfg, logger=logger)
    results = grid_search.run(forecasts, features, prices)

    output_path = output_dir / "summary.csv"
    results.to_csv(output_path, index=False)

    typer.echo(f"Grid search results written to {output_path}")


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
