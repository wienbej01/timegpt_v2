from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer
import yaml

from timegpt_v2.backtest.aggregation import (
    PhaseConfig,
    assign_phases,
    compute_cost_scenarios,
    compute_portfolio_summaries,
)
from timegpt_v2.backtest.grid import GridSearch
from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.eval.calibration import reliability_curve
from timegpt_v2.eval.metrics_forecast import (
    mae,
    pinball_loss,
    pit_coverage,
    rmae,
    rmse,
    rrmse,
)
from timegpt_v2.eval.metrics_trading import hit_rate, max_drawdown, sharpe_ratio
from timegpt_v2.fe.base_features import build_feature_matrix
from timegpt_v2.fe.context import FeatureContext
from timegpt_v2.forecast.scheduler import ForecastScheduler, get_trading_holidays
from timegpt_v2.forecast.timegpt_client import TimeGPTClient, TimeGPTConfig
from timegpt_v2.framing.build_payloads import build_x_df_for_horizon, build_y_df
from timegpt_v2.io.gcs_reader import GCSReader, ReaderConfig
from timegpt_v2.quality.checks import DataQualityChecker
from timegpt_v2.quality.contracts import DataQualityPolicy
from timegpt_v2.reports.builder import build_report
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import RuleParams, TradingRules
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


def _configure_logger(log_path: Path, *, name: str = "timegpt_v2.cli") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _iter_chunks(items: Sequence[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


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

    active_windows_cfg = forecast_cfg.get("active_windows", [])
    active_windows: list[tuple[time, time]] = []
    for window in active_windows_cfg:
        try:
            start_str = window["start"]
            end_str = window["end"]
        except (KeyError, TypeError) as exc:  # pragma: no cover - config error
            raise typer.BadParameter("active_windows entries must include start/end") from exc
        try:
            start_time = datetime.strptime(str(start_str), "%H:%M").time()
            end_time = datetime.strptime(str(end_str), "%H:%M").time()
        except ValueError as exc:  # pragma: no cover - config error
            raise typer.BadParameter(f"Invalid active window: {window}") from exc
        if end_time <= start_time:
            raise typer.BadParameter("active window end must be after start")
        active_windows.append((start_time, end_time))

    max_snapshots_per_day = forecast_cfg.get("max_snapshots_per_day")
    if max_snapshots_per_day is not None:
        max_snapshots_per_day = int(max_snapshots_per_day)
    max_snapshots_total = forecast_cfg.get("max_snapshots_total")
    if max_snapshots_total is not None:
        max_snapshots_total = int(max_snapshots_total)

    trade_dates = sorted(features["timestamp_et"].dt.date.unique())
    holidays = get_trading_holidays(years=sorted(list(set(d.year for d in trade_dates))))
    scheduler = ForecastScheduler(
        dates=trade_dates,
        snapshots=snapshot_times,
        tz=zone,
        holidays=holidays,
        active_windows=active_windows,
        max_snapshots_per_day=max_snapshots_per_day,
        max_total_snapshots=max_snapshots_total,
    )
    snapshots = scheduler.generate_snapshots()

    logger = _configure_logger(logs_dir / "forecast.log", name="timegpt_v2.forecast")
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
    max_batch_size_raw = int(forecast_cfg.get("max_batch_size", 0))
    max_batch_size = max_batch_size_raw if max_batch_size_raw > 0 else len(expected_symbols)
    max_batch_size = max(max_batch_size, 1)

    for snapshot_local in snapshots:
        snapshot_utc = pd.Timestamp(snapshot_local).tz_convert("UTC")
        history = build_y_df(features, snapshot_utc)
        if history.empty:
            continue
        latest = history.groupby("unique_id")["ds"].max()
        if not (latest == snapshot_utc).all():
            continue
        available_ids = sorted(str(uid) for uid in latest.index)

        for chunk_ids in _iter_chunks(available_ids, max_batch_size):
            history_chunk = history[history["unique_id"].isin(chunk_ids)].copy()
            future_chunk = build_x_df_for_horizon(
                features,
                snapshot_utc,
                horizon,
                symbols=chunk_ids,
            )
            forecast_df = client.forecast(
                history_chunk,
                future_chunk,
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
    run_dir = Path("artifacts") / "runs" / run_id
    logs_dir = run_dir / "logs"
    trades_dir = run_dir / "trades"
    eval_dir = run_dir / "eval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    forecasts_path = run_dir / "forecasts" / "quantiles.csv"
    features_path = run_dir / "features" / "features.parquet"
    prices_path = run_dir / "validation" / "clean.parquet"
    trading_cfg_path = config_dir / "trading.yaml"
    backtest_cfg_path = config_dir / "backtest.yaml"

    for path, msg in (
        (forecasts_path, "Forecasts not found. Run `forecast` first."),
        (features_path, "Features not found. Run `build-features` first."),
        (prices_path, "Validated prices not found. Run `check-data` first."),
        (trading_cfg_path, "Trading config missing."),
        (backtest_cfg_path, "Backtest config missing."),
    ):
        if not path.exists():
            raise typer.BadParameter(msg)

    logger = _configure_logger(logs_dir / "backtest.log", name="timegpt_v2.backtest")

    forecasts = pd.read_csv(forecasts_path)
    features = pd.read_parquet(features_path)
    prices = pd.read_parquet(prices_path)
    if "timestamp" not in prices.columns:
        raise typer.BadParameter("Prices file must include a `timestamp` column.")
    if "close" not in prices.columns:
        if "adj_close" in prices.columns:
            prices = prices.rename(columns={"adj_close": "close"})
        else:
            raise typer.BadParameter("Prices file must include `close` or `adj_close` column.")

    trading_cfg = _load_yaml(trading_cfg_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    tick_size = float(trading_cfg.get("tick_size", 0.01))
    phase_cfg = PhaseConfig.from_mapping(  # type: ignore[arg-type]
        {
            "in_sample": backtest_cfg.get("in_sample_months", []) or [],
            "oos": backtest_cfg.get("oos_months", []) or [],
            "stress": backtest_cfg.get("stress_months", []) or [],
        }
    )
    portfolio_aggregation = str(backtest_cfg.get("portfolio_aggregation", "equal_weight"))

    def _select_param(values: Any) -> float:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if not values:
                raise typer.BadParameter("Trading grid values cannot be empty.")
            return float(values[0])
        return float(values)

    params = RuleParams(
        k_sigma=_select_param(trading_cfg.get("k_sigma")),
        s_stop=_select_param(trading_cfg.get("s_stop")),
        s_take=_select_param(trading_cfg.get("s_take")),
    )

    trading_costs = TradingCosts(
        fee_bps=float(trading_cfg.get("fees_bps", 0.0)),
        half_spread_ticks=trading_cfg.get("half_spread_ticks", {}),
    )
    time_stop = datetime.strptime(str(trading_cfg.get("time_stop_et", "15:55")), "%H:%M").time()
    rules = TradingRules(
        costs=trading_costs,
        time_stop=time_stop,
        daily_trade_cap=int(trading_cfg.get("daily_trade_cap", 1)),
        max_open_per_symbol=int(trading_cfg.get("max_open_per_symbol", 1)),
    )
    simulator = BacktestSimulator(
        rules=rules,
        params=params,
        logger=logger,
        tick_size=tick_size,
    )

    trades_df, summary_df = simulator.run(forecasts, features, prices)
    trades_df = assign_phases(trades_df, config=phase_cfg)
    if "trade_month" in trades_df.columns:
        trades_df.drop(columns=["trade_month"], inplace=True)

    portfolio_summary, symbol_summary = compute_portfolio_summaries(
        trades_df,
        aggregation=portfolio_aggregation,
    )

    trades_path = trades_dir / "bt_trades.csv"
    summary_path = eval_dir / "bt_summary.csv"
    portfolio_path = eval_dir / "portfolio_summary.csv"
    per_symbol_path = eval_dir / "per_symbol_summary.csv"

    trades_output = trades_df.copy()
    for column in ("entry_ts", "exit_ts"):
        trades_output[column] = pd.to_datetime(trades_output[column], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    trades_output.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    portfolio_summary.to_csv(portfolio_path, index=False)
    symbol_summary.to_csv(per_symbol_path, index=False)

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.setdefault("steps", {})
    meta["command"] = "backtest"
    meta["steps"]["backtest"] = {
        "run_id": run_id,
        "trades_path": str(trades_path),
        "summary_path": str(summary_path),
        "portfolio_summary_path": str(portfolio_path),
        "per_symbol_summary_path": str(per_symbol_path),
        "trade_count": int(summary_df.iloc[0]["trade_count"]),
        "total_net_pnl": float(summary_df.iloc[0]["total_net_pnl"]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _append_log(
        logs_dir / "events.log",
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "backtest",
            "run_id": run_id,
            "trade_count": int(summary_df.iloc[0]["trade_count"]),
        },
    )

    typer.echo(f"Trades written to {trades_path}")
    typer.echo(f"Summary written to {summary_path}")


@app.command()
def evaluate(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Evaluate model and trading performance metrics."""
    run_dir = Path("artifacts") / "runs" / run_id
    forecasts_path = run_dir / "forecasts" / "quantiles.csv"
    trades_path = run_dir / "trades" / "bt_trades.csv"
    portfolio_path = run_dir / "eval" / "portfolio_summary.csv"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not forecasts_path.exists():
        raise typer.BadParameter("Forecasts not found. Run `forecast` first.")
    if not trades_path.exists():
        raise typer.BadParameter("Trades not found. Run `backtest` first.")
    if not portfolio_path.exists():
        raise typer.BadParameter("Portfolio summary not found. Run `backtest` first.")

    backtest_cfg_path = config_dir / "backtest.yaml"
    if not backtest_cfg_path.exists():
        raise typer.BadParameter("Backtest config missing.")
    backtest_cfg = _load_yaml(backtest_cfg_path)
    cost_multipliers = backtest_cfg.get("cost_multipliers", [1.0, 1.5, 2.0])
    portfolio_aggregation = str(backtest_cfg.get("portfolio_aggregation", "equal_weight"))

    forecasts = pd.read_csv(forecasts_path)
    required_forecast_cols = {"symbol", "q25", "q50", "q75", "y_true", "ts_utc"}
    missing_forecast_cols = required_forecast_cols - set(forecasts.columns)
    if missing_forecast_cols:
        raise typer.BadParameter(
            f"Forecasts file missing required columns: {sorted(missing_forecast_cols)}"
        )
    forecasts["ts_utc"] = pd.to_datetime(forecasts["ts_utc"], utc=True, errors="coerce")
    if forecasts["ts_utc"].isna().any():
        raise typer.BadParameter("Forecasts contain invalid timestamps in `ts_utc`.")

    trades = pd.read_csv(trades_path)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")

    portfolio_summary = pd.read_csv(portfolio_path)

    # Forecast evaluation per symbol
    forecast_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for symbol, group in forecasts.sort_values("ts_utc").groupby("symbol", sort=True):
        y_true = group["y_true"].astype(float)
        q50 = group["q50"].astype(float)
        q25 = group["q25"].astype(float)
        q75 = group["q75"].astype(float)
        y_persistence = y_true.shift(1)
        valid_mask = y_persistence.notna()
        if valid_mask.sum() == 0:
            continue

        yt = y_true[valid_mask]
        yp = q50[valid_mask]
        ypersistence = y_persistence[valid_mask].astype(float)
        q25_valid = q25[valid_mask]
        q75_valid = q75[valid_mask]

        forecast_rows.append(
            {
                "symbol": symbol,
                "count": int(len(yt)),
                "mae": float(mae(yt, yp)),
                "rmse": float(rmse(yt, yp)),
                "rmae": float(rmae(yt, yp, ypersistence)),
                "rrmse": float(rrmse(yt, yp, ypersistence)),
                "pinball_loss_q25": float(pinball_loss(yt, q25_valid, 0.25)),
                "pinball_loss_q75": float(pinball_loss(yt, q75_valid, 0.75)),
                "pit_coverage": float(pit_coverage(yt, q25_valid, q75_valid)),
            }
        )

        binary_actual = (y_true <= q50).astype(float)
        prob_pred = np.full(len(binary_actual), 0.5, dtype=float)
        bin_pred, bin_true = reliability_curve(binary_actual, prob_pred, n_bins=10)
        for idx, (pred_val, true_val) in enumerate(zip(bin_pred, bin_true, strict=False)):
            reliability_rows.append(
                {
                    "symbol": symbol,
                    "bin": idx,
                    "prob_pred": float(pred_val),
                    "prob_true": float(true_val),
                }
            )

    if not forecast_rows:
        raise typer.BadParameter(
            "Forecasts do not contain enough data per symbol to compute metrics."
        )

    forecast_metrics_df = pd.DataFrame(forecast_rows)
    forecast_metrics_path = eval_dir / "forecast_metrics.csv"
    forecast_metrics_df.to_csv(forecast_metrics_path, index=False)
    typer.echo(f"Forecast metrics written to {forecast_metrics_path}")

    reliability_df = pd.DataFrame(reliability_rows)
    reliability_path = eval_dir / "pit_reliability.csv"
    reliability_df.to_csv(reliability_path, index=False)

    median_rmae = float(forecast_metrics_df["rmae"].median())
    median_rrmse = float(forecast_metrics_df["rrmse"].median())
    median_pit = float(forecast_metrics_df["pit_coverage"].median())
    total_obs = int(forecast_metrics_df["count"].sum())
    coverage_tolerance = 0.02 if total_obs >= 50 else 0.1

    # Trading evaluation
    net_pnl_series = trades.get("net_pnl")
    if net_pnl_series is None:
        raise typer.BadParameter("Trades file missing `net_pnl` column.")
    net_pnl_series = trades.sort_values("entry_ts").set_index("entry_ts")["net_pnl"].astype(float)
    trading_metrics = {
        "sharpe_ratio": sharpe_ratio(net_pnl_series, trading_days=252),
        "max_drawdown": max_drawdown(net_pnl_series),
        "hit_rate": hit_rate(net_pnl_series),
        "total_pnl": net_pnl_series.sum(),
    }
    trading_metrics_df = pd.DataFrame([trading_metrics])
    trading_metrics_path = eval_dir / "bt_summary.csv"
    trading_metrics_df.to_csv(trading_metrics_path, index=False)
    typer.echo(f"Trading metrics written to {trading_metrics_path}")

    cost_table = compute_cost_scenarios(
        trades,
        multipliers=cost_multipliers,
        aggregation=portfolio_aggregation,
    )
    cost_sensitivity_path = eval_dir / "cost_sensitivity.csv"
    cost_table.to_csv(cost_sensitivity_path, index=False)
    typer.echo(f"Cost sensitivity written to {cost_sensitivity_path}")

    # Gates
    if median_rmae >= 0.95 or median_rrmse >= 0.97:
        typer.echo("Forecast evaluation gates failed.")
        raise typer.Exit(code=1)

    if abs(median_pit - 0.5) > coverage_tolerance:
        typer.echo("Calibration gate failed.")
        raise typer.Exit(code=1)

    oos_row = portfolio_summary[
        (portfolio_summary["phase"] == "oos") & (portfolio_summary["level"] == "portfolio")
    ]
    if oos_row.empty:
        typer.echo("Missing OOS portfolio metrics.")
        raise typer.Exit(code=1)
    oos_metrics = oos_row.iloc[0]
    if (
        oos_metrics.get("sharpe", 0.0) < 0.5
        or oos_metrics.get("total_net_pnl", 0.0) <= 0.0
        or oos_metrics.get("hit_rate", 0.0) < 0.48
    ):
        typer.echo("OOS portfolio gates failed.")
        raise typer.Exit(code=1)

    c15 = cost_table.loc[(cost_table["cost_multiplier"] - 1.5).abs() < 1e-6]
    if c15.empty or c15.iloc[0]["total_net_pnl"] < 0.0:
        typer.echo("Cost sensitivity gate failed at 1.5× costs.")
        raise typer.Exit(code=1)

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    steps = meta.setdefault("steps", {})
    evaluate_meta = steps.setdefault("evaluate", {})
    meta["command"] = "evaluate"
    evaluate_meta.update(
        {
            "run_id": run_id,
            "forecast_metrics_path": str(forecast_metrics_path),
            "trading_metrics_path": str(trading_metrics_path),
            "cost_sensitivity_path": str(cost_sensitivity_path),
            "reliability_path": str(reliability_path),
            "median_rmae": median_rmae,
            "median_rrmse": median_rrmse,
            "median_pit_coverage": median_pit,
            "forecast_gate_pass": median_rmae < 0.95 and median_rrmse < 0.97,
            "calibration_gate_pass": abs(median_pit - 0.5) <= coverage_tolerance,
            "coverage_tolerance": coverage_tolerance,
            "total_forecast_observations": total_obs,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    typer.echo("Evaluation gates passed.")


@app.command()
def report(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
) -> None:
    """Assemble final report for the run."""
    run_dir = Path("artifacts") / "runs" / run_id
    report_path = build_report(run_dir, config_dir=config_dir)

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.setdefault("steps", {})
    meta.setdefault("steps", {}).setdefault("report", {})
    meta["command"] = "report"
    meta["steps"]["report"].update({"report_path": str(report_path)})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    typer.echo(f"Robustness report written to {report_path}")


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

    logger = _configure_logger(run_dir / "logs" / "sweep.log", name="timegpt_v2.sweep")
    tick_size = float(trading_cfg.get("tick_size", 0.01))
    grid_search = GridSearch(
        trading_cfg=trading_cfg,
        logger=logger,
        output_root=output_dir,
        tick_size=tick_size,
    )
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
