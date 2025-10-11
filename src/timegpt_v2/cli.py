from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer
import yaml

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # Fallback simple .env parser
    env_path = Path(".env")
    if env_path.exists():
        for _line in env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k, _v)

from timegpt_v2.backtest.aggregation import (
    PhaseConfig,
    assign_phases,
    compute_cost_scenarios,
    compute_portfolio_summaries,
)
from timegpt_v2.backtest.grid import GridSearch
from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.eval.calibration import (
    CalibrationConfig,
    ForecastCalibrator,
    apply_conformal_widening,
    compute_embargo_cutoff,
    enforce_quantile_monotonicity,
    filter_calibration_window,
    reliability_curve,
)
from timegpt_v2.eval.metrics_forecast import (
    interval_width_stats,
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
from timegpt_v2.forecast.scaling import TargetScaler, TargetScalingConfig
from timegpt_v2.forecast.scheduler import ForecastScheduler, get_trading_holidays
from timegpt_v2.forecast.sweep import ForecastGridSearch, ForecastGridSpec
from timegpt_v2.forecast.timegpt_client import (
    NixtlaTimeGPTBackend,
    TimeGPTClient,
    TimeGPTConfig,
    TimeGPTRetryPolicy,
    _LocalDeterministicBackend,
)
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


def _coerce_optional_int(value: object | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is subclass of int but rarely intended
        raise typer.BadParameter(f"{field} must be integer-like, not boolean")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(stripped)
        except ValueError as exc:  # pragma: no cover - config error
            raise typer.BadParameter(f"{field} must be integer-like") from exc
    raise typer.BadParameter(f"{field} must be integer-like")


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
    universe_config: str = typer.Option(
        "universe.yaml", "--universe-config", help="Universe config file name"
    ),
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

    universe_cfg = _load_yaml(config_dir / universe_config)
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
    skip_ts_norm = bool(gcs_cfg.get("skip_timestamp_normalization", False))
    if not template:
        raise typer.BadParameter("configs/data.yaml must define gcs.template")

    reader = GCSReader(
        ReaderConfig(bucket=bucket, template=template, skip_timestamp_normalization=skip_ts_norm)
    )

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
    horizon_base = _coerce_optional_int(forecast_cfg.get("horizon_min", 15), field="horizon_min")
    horizon = horizon_base if horizon_base is not None else 15
    freq = str(forecast_cfg.get("freq", "min"))
    tz_name = str(forecast_cfg.get("tz", "America/New_York"))
    snapshot_presets_cfg = forecast_cfg.get("snapshot_presets")
    preset_name: str | None = None
    preset_active_windows_cfg: object | None = None
    preset_max_per_day: object | None = None
    preset_max_total: object | None = None

    if snapshot_presets_cfg:
        if not isinstance(snapshot_presets_cfg, Sequence):
            raise typer.BadParameter("snapshot_presets must be a sequence")
        presets: dict[str, dict[str, object]] = {}
        default_name: str | None = None
        for entry in snapshot_presets_cfg:
            if not isinstance(entry, Mapping):
                raise typer.BadParameter("Each snapshot preset must be a mapping")
            entry_name_raw = entry.get("name")
            if entry_name_raw is None:
                raise typer.BadParameter("Each snapshot preset must include a name")
            entry_name = str(entry_name_raw)
            presets[entry_name] = dict(entry)
            if default_name is None:
                default_name = entry_name
        if default_name is None:
            raise typer.BadParameter("snapshot_presets cannot be empty")
        preset_name_raw = forecast_cfg.get("snapshot_preset", default_name)
        preset_name = str(preset_name_raw)
        if preset_name not in presets:
            available = ", ".join(sorted(presets))
            raise typer.BadParameter(
                f"snapshot_preset '{preset_name}' not found. Available presets: {available}"
            )
        selected_preset = presets[preset_name]
        snapshot_values_raw = selected_preset.get("times")
        if not isinstance(snapshot_values_raw, Sequence):
            raise typer.BadParameter(f"snapshot preset '{preset_name}' times must be a sequence")
        snapshot_strs = [str(item) for item in snapshot_values_raw]
        if not snapshot_strs:
            raise typer.BadParameter(f"snapshot preset '{preset_name}' must define times")
        preset_horizon = _coerce_optional_int(
            selected_preset.get("horizon_min"), field="horizon_min"
        )
        if preset_horizon is not None:
            horizon = preset_horizon
        preset_active_windows_cfg = selected_preset.get("active_windows")
        preset_max_per_day = selected_preset.get("max_snapshots_per_day")
        preset_max_total = selected_preset.get("max_total_snapshots")
    else:
        snapshot_strs = forecast_cfg.get("snapshots_et", [])
        if not snapshot_strs:
            raise typer.BadParameter("forecast.yaml must define snapshots_et or snapshot_presets")
        if not isinstance(snapshot_strs, Sequence):
            raise typer.BadParameter("snapshots_et must be a sequence")
        snapshot_strs = [str(item) for item in snapshot_strs]

    target_cfg_raw = forecast_cfg.get("target")
    if target_cfg_raw is not None and not isinstance(target_cfg_raw, Mapping):
        raise typer.BadParameter("forecast target configuration must be a mapping")
    target_scaling_config = TargetScalingConfig.from_mapping(target_cfg_raw)
    scaler = TargetScaler(target_scaling_config)

    retry_cfg = forecast_cfg.get("retry", {})
    retry_policy = TimeGPTRetryPolicy()
    if isinstance(retry_cfg, Mapping):
        retry_policy = TimeGPTRetryPolicy(
            max_attempts=int(retry_cfg.get("max", retry_policy.max_attempts)),
            backoff_seconds=float(retry_cfg.get("backoff_sec", retry_policy.backoff_seconds)),
        )
    timeout_sec = float(forecast_cfg.get("timeout_sec", 30.0))
    base_url_raw = forecast_cfg.get("base_url")
    base_url = str(base_url_raw).strip() if base_url_raw is not None else ""
    base_url = base_url or None
    api_key_cfg = forecast_cfg.get("api_key")
    backend_mode = str(forecast_cfg.get("backend", "auto")).lower()

    zone = ZoneInfo(tz_name)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features_et = features["timestamp"].dt.tz_convert(zone)
    features = features.assign(timestamp_et=features_et)

    skip_events_cfg = forecast_cfg.get("skip_events", [])
    if isinstance(skip_events_cfg, str):
        skip_event_names = [skip_events_cfg]
    elif isinstance(skip_events_cfg, Sequence):
        skip_event_names = [str(name) for name in skip_events_cfg]
    elif skip_events_cfg is None:
        skip_event_names = []
    else:
        raise typer.BadParameter("skip_events must be a string or a sequence of strings")

    skip_dates: set[date] = set()
    if skip_event_names:
        timestamp_et = features["timestamp_et"]
        for event_name in skip_event_names:
            if event_name not in features.columns:
                typer.echo(
                    f"Skip event column '{event_name}' not found in features.",
                    err=True,
                )
                continue
            mask = features[event_name].astype(bool)
            if mask.any():
                event_dates = timestamp_et.loc[mask].dt.date
                skip_dates.update(event_dates.tolist())

    snapshot_times: list[time] = []
    for snapshot in snapshot_strs:
        try:
            parsed = datetime.strptime(str(snapshot), "%H:%M").time()
        except ValueError as exc:  # pragma: no cover - config error
            raise typer.BadParameter(f"Invalid snapshot time: {snapshot}") from exc
        snapshot_times.append(parsed)

    active_windows_cfg_raw = (
        preset_active_windows_cfg
        if preset_active_windows_cfg is not None
        else forecast_cfg.get("active_windows", [])
    )
    if active_windows_cfg_raw is None:
        active_windows_cfg_list: list[Mapping[str, object]] = []
    elif isinstance(active_windows_cfg_raw, Sequence) and not isinstance(
        active_windows_cfg_raw, (str, bytes)
    ):
        active_windows_cfg_list = [
            window for window in active_windows_cfg_raw if window is not None
        ]
    elif isinstance(active_windows_cfg_raw, Mapping):
        raise typer.BadParameter("active_windows must be a sequence of mappings, not a mapping")
    else:
        raise typer.BadParameter("active_windows must be a sequence of mappings")

    active_windows_cfg = active_windows_cfg_list
    active_windows: list[tuple[time, time]] = []
    for window in active_windows_cfg:
        if not isinstance(window, Mapping):
            raise typer.BadParameter("active_windows entries must be mappings with start/end")
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

    if preset_max_per_day is not None:
        max_snapshots_per_day = _coerce_optional_int(
            preset_max_per_day, field="max_snapshots_per_day"
        )
    else:
        max_snapshots_per_day = _coerce_optional_int(
            forecast_cfg.get("max_snapshots_per_day"), field="max_snapshots_per_day"
        )

    if preset_max_total is not None:
        max_snapshots_total = _coerce_optional_int(preset_max_total, field="max_total_snapshots")
    else:
        max_snapshots_total = _coerce_optional_int(
            forecast_cfg.get("max_total_snapshots"), field="max_total_snapshots"
        )

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
        skip_dates=sorted(skip_dates),
    )
    snapshots = scheduler.generate_snapshots()

    logger = _configure_logger(logs_dir / "forecast.log", name="timegpt_v2.forecast")
    cache = ForecastCache(forecasts_dir / "cache", logger=logger)
    backend: NixtlaTimeGPTBackend | None = None
    if backend_mode not in {"auto", "nixtla", "stub"}:
        raise typer.BadParameter("forecast.yaml backend must be one of: auto, nixtla, stub")
    resolved_api_key = ""
    if isinstance(api_key_cfg, str):
        resolved_api_key = api_key_cfg.strip()
    elif api_key_cfg is not None:
        resolved_api_key = str(api_key_cfg).strip()
    if backend_mode in {"auto", "nixtla", "stub"}:
        if backend_mode == "stub":
            # Use local deterministic backend for testing
            backend = _LocalDeterministicBackend()
        elif not resolved_api_key:
            resolved_api_key = os.environ.get("TIMEGPT_API_KEY", "") or os.environ.get(
                "NIXTLA_API_KEY", ""
            )
        if backend_mode in {"auto", "nixtla"} and (resolved_api_key or backend_mode == "nixtla"):
            try:
                backend = NixtlaTimeGPTBackend(
                    api_key=resolved_api_key or None,
                    base_url=base_url,
                    retry=retry_policy,
                    timeout=timeout_sec,
                )
            except Exception as exc:
                message = f"Unable to initialize TimeGPT backend: {exc}"
                if backend_mode == "nixtla":
                    raise typer.BadParameter(message) from exc
                logger.warning("%s Falling back to deterministic backend.", message)

    # Allow stub backend for testing
    if backend is None:
        raise typer.BadParameter(
            "TimeGPT backend required. "
            "Ensure TIMEGPT_API_KEY is set in environment or .env, "
            "or set backend: stub in forecast.yaml for testing."
        )

    # Backend-mode assertion and monitoring
    if backend_mode not in {"nixtla", "stub", "auto"}:
        logger.error("Backend mode assertion failed: expected 'nixtla', got '%s'", backend_mode)
        raise typer.BadParameter("Backend mode must be 'nixtla' or 'stub' for production runs")
    backend_name = backend.__class__.__name__
    logger.info(
        "Using forecast backend=%s (mode=%s) horizon=%s freq=%s quantiles=%s preset=%s",
        backend_name,
        backend_mode,
        horizon,
        freq,
        list(quantiles),
        preset_name,
    )
    client = TimeGPTClient(
        backend=backend,
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
        history = build_y_df(
            features,
            snapshot_utc,
            target_column=scaler.target_column,
            rolling_window_days=90,  # Use 90-day rolling window to reduce payload size
        )
        if history.empty:
            continue
        latest = history.groupby("unique_id")["ds"].max()
        if not (latest == snapshot_utc).all():
            continue
        available_ids = sorted(str(uid) for uid in latest.index)
        # Skip snapshots with insufficient history for TimeGPT prediction intervals
        min_samples = 25
        symbol_counts = history.groupby("unique_id").size()
        if (symbol_counts < min_samples).any():
            logger.warning(
                f"Skipping snapshot {snapshot_utc}: insufficient history (< {min_samples} samples for symbols: "
                f"{sorted(symbol_counts[symbol_counts < min_samples].index.tolist())}"
            )
            continue

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
    combined = scaler.inverse_quantiles(
        combined,
        features=features,
        quantile_columns=quantile_cols,
    )
    pre_calibration = combined[["forecast_ts", "unique_id", *quantile_cols]].copy()
    pre_calibration.rename(
        columns={col: f"{col}_pre_calib" for col in quantile_cols},
        inplace=True,
    )

    calibration_cfg_raw = forecast_cfg.get("calibration")
    calibration_config: CalibrationConfig | None = None
    calibrator_loaded = False
    conformal_applied = False
    if calibration_cfg_raw is not None:
        calibration_config = CalibrationConfig.from_mapping(calibration_cfg_raw)
        calibrator = ForecastCalibrator(calibration_config)
        calibrator_loaded = calibrator.load()
        if calibrator_loaded:
            logger.info("Applying calibration models from %s", calibration_config.model_path)
            combined = calibrator.apply(combined)
            logger.info("Calibration applied successfully")
        else:
            logger.warning(
                "Calibration models not found at %s, skipping calibration",
                calibration_config.model_path,
            )

    combined = enforce_quantile_monotonicity(combined, logger=logger)

    for _snapshot_ts, group in combined.groupby("snapshot_utc"):
        group_symbols = set(group["unique_id"].astype(str))
        if group_symbols != set(expected_symbols):
            raise typer.Exit(code=1)
        for column in quantile_cols:
            if group[column].isna().any():
                raise typer.Exit(code=1)

    snapshot_values = sorted(ts.isoformat() for ts in combined["snapshot_utc"].unique())

    output = combined.rename(columns={"unique_id": "symbol", "forecast_ts": "ts_utc"})
    output["symbol"] = output["symbol"].astype(str)
    output["ts_utc"] = pd.to_datetime(output["ts_utc"], utc=True)
    if not pre_calibration.empty:
        pre_merge = pre_calibration.rename(columns={"forecast_ts": "ts_utc", "unique_id": "symbol"})
        pre_merge["symbol"] = pre_merge["symbol"].astype(str)
        pre_merge["ts_utc"] = pd.to_datetime(pre_merge["ts_utc"], utc=True)
        output = output.merge(pre_merge, on=["ts_utc", "symbol"], how="left")
    output.sort_values(["ts_utc", "symbol"], inplace=True)

    label_column = scaler.label_timestamp_column
    target_column = scaler.target_column
    missing_actual_columns = {label_column, "symbol", target_column} - set(features.columns)
    if missing_actual_columns:
        missing_list = ", ".join(sorted(missing_actual_columns))
        raise typer.BadParameter(
            f"Feature matrix missing required columns for targets: {missing_list}. "
            "Ensure feature generation is up to date with target configuration."
        )
    actuals = (
        features[[label_column, "symbol", target_column]]
        .dropna(subset=[label_column])
        .rename(columns={label_column: "ts_utc", target_column: "y_true"})
    )
    actuals["ts_utc"] = pd.to_datetime(actuals["ts_utc"], utc=True)
    output = output.merge(actuals, on=["ts_utc", "symbol"], how="left")

    missing_actuals = int(output["y_true"].isna().sum())
    if missing_actuals:
        logger.warning("Dropping %s forecast rows without ground-truth targets.", missing_actuals)
        output = output[output["y_true"].notna()]

    if output.empty:
        raise typer.Exit(code=1)

    if calibration_config and calibration_config.conformal_fallback:
        actuals_for_conformal = output[["symbol", "ts_utc", "y_true"]]
        if not actuals_for_conformal.empty:
            widened = apply_conformal_widening(
                output,
                actuals=actuals_for_conformal,
                pit_deviation_threshold=calibration_config.pit_deviation_threshold,
                window=calibration_config.conformal_window,
            )
            # If widening returns
            if not widened.empty:
                output = widened
                output = enforce_quantile_monotonicity(output, logger=logger)
                conformal_applied = True
            else:
                logger.warning("Conformal widening produced no rows; skipping fallback adjustment.")
        else:
            logger.warning("Skipping conformal fallback because no actuals were available.")

    ts_strings = output["ts_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output.loc[:, "ts_utc"] = ts_strings.to_numpy(dtype=object)
    pre_quantile_cols = [col for col in output.columns if col.endswith("_pre_calib")]
    ordered_columns = ["ts_utc", "symbol", *quantile_cols, *pre_quantile_cols, "y_true"]
    output = output[ordered_columns]
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
        "target_scaling": scaler.metadata,
        "snapshot_preset": preset_name,
        "horizon_minutes": horizon,
        "skip_events": skip_event_names,
        "calibration_applied": calibrator_loaded,
        "conformal_fallback_configured": (
            bool(calibration_config.conformal_fallback) if calibration_config else False
        ),
        "conformal_fallback_applied": conformal_applied,
        "skip_event_dates": sorted(d.isoformat() for d in skip_dates),
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

    # Check if summary_df has data before accessing iloc[0]
    if summary_df.empty:
        raise typer.BadParameter("Backtest produced no summary statistics.")

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
            "trade_count": int(summary_df.iloc[0]["trade_count"]) if not summary_df.empty else 0,
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
        interval_mean, interval_median = interval_width_stats(
            q25_valid.to_numpy(), q75_valid.to_numpy()
        )
        pit_post = float(pit_coverage(yt, q25_valid, q75_valid))
        row: dict[str, object] = {
            "symbol": symbol,
            "count": int(len(yt)),
            "mae": float(mae(yt, yp)),
            "rmse": float(rmse(yt, yp)),
            "rmae": float(rmae(yt, yp, ypersistence)),
            "rrmse": float(rrmse(yt, yp, ypersistence)),
            "pinball_loss_q25": float(pinball_loss(yt, q25_valid, 0.25)),
            "pinball_loss_q75": float(pinball_loss(yt, q75_valid, 0.75)),
            "pit_coverage": pit_post,
            "interval_width_mean": interval_mean,
            "interval_width_median": interval_median,
        }
        pre_cols = {"q25_pre_calib", "q50_pre_calib", "q75_pre_calib"}
        if pre_cols.issubset(group.columns):
            q25_pre = group["q25_pre_calib"].astype(float)[valid_mask]
            q75_pre = group["q75_pre_calib"].astype(float)[valid_mask]
            coverage_pre = float(pit_coverage(yt, q25_pre, q75_pre))
            row["pit_coverage_pre_calib"] = coverage_pre
            row["pit_coverage_delta"] = pit_post - coverage_pre
        forecast_rows.append(row)

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

    diagnostics_path = eval_dir / "forecast_diagnostics.csv"
    forecast_metrics_df.to_csv(diagnostics_path, index=False)
    typer.echo(f"Forecast diagnostics written to {diagnostics_path}")

    reliability_df = pd.DataFrame(reliability_rows)
    reliability_path = eval_dir / "pit_reliability.csv"
    reliability_df.to_csv(reliability_path, index=False)

    median_rmae = float(forecast_metrics_df["rmae"].median())
    median_rrmse = float(forecast_metrics_df["rrmse"].median())
    median_pit = float(forecast_metrics_df["pit_coverage"].median())
    median_pit_pre: float | None = None
    if "pit_coverage_pre_calib" in forecast_metrics_df.columns:
        median_pit_pre = float(forecast_metrics_df["pit_coverage_pre_calib"].median())
    total_obs = int(forecast_metrics_df["count"].sum())
    coverage_tolerance = 0.02 if total_obs >= 50 else 0.1
    min_obs_for_gates = 200

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
    if total_obs >= min_obs_for_gates:
        gate_failures: list[str] = []
        if median_rmae >= 0.95 or median_rrmse >= 0.97:
            gate_failures.append(f"rMAE/rRMSE ({median_rmae:.3f}/{median_rrmse:.3f}) ≥ thresholds")

        pit_error = abs(median_pit - 0.5)
        if pit_error > coverage_tolerance:
            gate_failures.append(
                f"PIT deviation {pit_error:.3f} > tolerance {coverage_tolerance:.3f}"
            )

        if gate_failures:
            typer.echo(
                "Forecast gates failed: " + "; ".join(gate_failures) + f" (obs={total_obs})",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        typer.echo(
            f"Skipping forecast gates (only {total_obs} observations; need {min_obs_for_gates}).",
            err=True,
        )

    trade_count_by_phase = (
        portfolio_summary.groupby("phase")["trade_count"].sum().to_dict()
        if not portfolio_summary.empty
        else {}
    )
    oos_row = portfolio_summary[
        (portfolio_summary["phase"] == "oos") & (portfolio_summary["level"] == "portfolio")
    ]
    if oos_row.empty:
        typer.echo(
            f"Missing OOS portfolio metrics. Trade counts: {trade_count_by_phase}",
            err=True,
        )
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
    if c15.empty:
        typer.echo("Cost sensitivity gate failed: no 1.5× cost scenario found.")
        raise typer.Exit(code=1)
    if len(c15) == 0 or c15.iloc[0]["total_net_pnl"] < 0.0:
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
            "forecast_diagnostics_path": str(diagnostics_path),
            "trading_metrics_path": str(trading_metrics_path),
            "cost_sensitivity_path": str(cost_sensitivity_path),
            "reliability_path": str(reliability_path),
            "median_rmae": median_rmae,
            "median_rrmse": median_rrmse,
            "median_pit_coverage": median_pit,
            "median_pit_coverage_pre_calib": median_pit_pre,
            "median_pit_coverage_delta": (
                median_pit - median_pit_pre if median_pit_pre is not None else None
            ),
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
    forecast_grid: Path | None = typer.Option(
        None,
        "--forecast-grid",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional forecast sweep specification (YAML).",
    ),
    execute: bool = typer.Option(
        True,
        "--execute/--plan-only",
        help="Execute pipeline for each forecast combo (default: execute).",
    ),
    reuse_baseline: bool = typer.Option(
        False,
        "--reuse-baseline/--no-reuse-baseline",
        help="Reuse artifacts from baseline run_id provided via --baseline-run for features/validation.",
    ),
    baseline_run: str | None = typer.Option(
        None,
        "--baseline-run",
        help="Baseline run_id to reuse artifacts from when --reuse-baseline is enabled.",
    ),
) -> None:
    """Execute trading parameter sweep."""
    run_dir = Path("artifacts") / "runs" / run_id
    output_dir = run_dir / "eval" / "grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _configure_logger(run_dir / "logs" / "sweep.log", name="timegpt_v2.sweep")
    trading_cfg_path = grid_config or config_dir / "trading.yaml"
    trading_cfg = _load_yaml(trading_cfg_path)

    forecasts_path = run_dir / "forecasts" / "quantiles.csv"
    features_path = run_dir / "features" / "features.parquet"
    prices_path = run_dir / "validation" / "clean.parquet"

    forecasts: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None

    if forecasts_path.exists() and features_path.exists() and prices_path.exists():
        forecasts = pd.read_csv(forecasts_path)
        features = pd.read_parquet(features_path)
        prices = pd.read_parquet(prices_path)

    if forecast_grid is not None:
        forecast_cfg_path = config_dir / "forecast.yaml"
        base_forecast_cfg = _load_yaml(forecast_cfg_path)
        spec_mapping = _load_yaml(forecast_grid)
        grid_spec = ForecastGridSpec.from_mapping(spec_mapping, base_config=base_forecast_cfg)

        def _invoke_forecast(**kwargs: object) -> None:
            forecast(config_dir=kwargs["config_dir"], run_id=kwargs["run_id"])  # type: ignore[arg-type]

        def _invoke_backtest(**kwargs: object) -> None:
            backtest(config_dir=kwargs["config_dir"], run_id=kwargs["run_id"])  # type: ignore[arg-type]

        def _invoke_evaluate(**kwargs: object) -> None:
            evaluate(config_dir=kwargs["config_dir"], run_id=kwargs["run_id"])  # type: ignore[arg-type]

        search = ForecastGridSearch(
            base_config_dir=config_dir,
            base_forecast_config=base_forecast_cfg,
            grid_spec=grid_spec,
            output_root=output_dir / "forecast_grid",
            run_id_prefix=f"{run_id}_fg",
            forecast_cmd=_invoke_forecast,
            backtest_cmd=_invoke_backtest,
            evaluate_cmd=_invoke_evaluate,
            logger=logger,
            baseline_run=baseline_run,
        )
        plan = search.run(execute=execute, reuse_baseline_artifacts=reuse_baseline)
        plan_path = output_dir / "forecast_grid_plan.csv"
        plan.to_csv(plan_path, index=False)
        typer.echo(f"Forecast grid plan written to {plan_path}")

    if forecasts is None or features is None or prices is None:
        if forecast_grid is None:
            raise typer.BadParameter(
                "Forecasts not found. Run `forecast` first or supply --forecast-grid plan."
            )
        return

    grid_output_dir = output_dir / "trading_grid"
    grid_output_dir.mkdir(parents=True, exist_ok=True)
    tick_size = float(trading_cfg.get("tick_size", 0.01))
    grid_search = GridSearch(
        trading_cfg=trading_cfg,
        logger=logger,
        output_root=grid_output_dir,
        tick_size=tick_size,
    )
    results = grid_search.run(forecasts, features, prices)

    output_path = grid_output_dir / "summary.csv"
    results.to_csv(output_path, index=False)
    typer.echo(f"Trading grid search results written to {output_path}")


@app.command()
def calibrate(
    config_dir: Path = CONFIG_DIR_OPTION,
    run_id: str = RUN_ID_OPTION,
    baseline_run: str | None = typer.Option(
        None,
        "--baseline-run",
        help="Baseline run_id to use for calibration training data. If not provided, uses current run_id.",
    ),
    embargo_days: int = typer.Option(
        1,
        "--embargo-days",
        help="Minimum trading days between calibration end and evaluation start.",
    ),
) -> None:
    """Fit forecast calibration models using historical forecasts vs actuals."""
    run_dir = Path("artifacts") / "runs" / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_id = baseline_run or run_id
    baseline_dir = Path("artifacts") / "runs" / baseline_run_id

    forecast_cfg = _load_yaml(config_dir / "forecast.yaml")
    backtest_cfg = _load_yaml(config_dir / "backtest.yaml")

    calibration_cfg_raw = forecast_cfg.get("calibration")
    if calibration_cfg_raw is None:
        raise typer.BadParameter("forecast.yaml must include calibration configuration")
    calibration_config = CalibrationConfig.from_mapping(calibration_cfg_raw)

    forecasts_path = baseline_dir / "forecasts" / "quantiles.csv"
    if not forecasts_path.exists():
        raise typer.BadParameter(f"Forecasts not found in baseline run {baseline_run_id}")

    forecasts = pd.read_csv(forecasts_path)
    if "ts_utc" not in forecasts.columns or "symbol" not in forecasts.columns:
        raise typer.BadParameter(
            "Forecasts must include 'ts_utc' and 'symbol' columns for calibration."
        )
    if not any(col.startswith("q") for col in forecasts.columns):
        raise typer.BadParameter("Forecasts must include quantile columns prefixed with 'q'.")

    forecasts["ts_utc"] = pd.to_datetime(forecasts["ts_utc"], utc=True, errors="coerce")
    if forecasts["ts_utc"].isna().any():
        raise typer.BadParameter("Forecasts contain invalid timestamps in 'ts_utc'.")

    if "y_true" not in forecasts.columns:
        raise typer.BadParameter("Forecasts must include 'y_true' column for calibration.")

    eval_start_months = backtest_cfg.get("oos_months", [])
    if not eval_start_months:
        raise typer.BadParameter("backtest.yaml must define oos_months for embargo calculation")

    eval_start_str = str(eval_start_months[0]) + "-01"
    eval_start = datetime.fromisoformat(eval_start_str + "T00:00:00").date()

    holidays = get_trading_holidays(years=sorted({eval_start.year, eval_start.year - 1}))
    embargo_cutoff = compute_embargo_cutoff(eval_start, embargo_days, holidays)
    window_days = (
        calibration_config.calibration_window_days
        if calibration_config.calibration_window_days > 0
        else None
    )
    calibration_subset, calibration_window_start = filter_calibration_window(
        forecasts,
        embargo_cutoff=embargo_cutoff,
        window_days=window_days,
    )
    if calibration_subset.empty:
        window_msg = (
            f" between {calibration_window_start} and {embargo_cutoff}"
            if calibration_window_start
            else f" on/before {embargo_cutoff}"
        )
        raise typer.BadParameter(
            f"No calibration forecasts available{window_msg}. "
            "Ensure prior runs exist or adjust calibration_window_days/embargo_days."
        )

    rows_in_window = len(calibration_subset)
    calibration_subset = calibration_subset.dropna(subset=["y_true"])
    rows_with_targets = len(calibration_subset)
    dropped_missing_targets = rows_in_window - rows_with_targets
    if rows_with_targets == 0:
        raise typer.BadParameter(
            "No calibration rows with y_true available after embargo/window filtering."
        )

    actuals_subset = calibration_subset[["symbol", "ts_utc", "y_true"]].copy()
    forecasts_subset = calibration_subset.drop(columns=["y_true"])

    logger = _configure_logger(logs_dir / "calibrate.log", name="timegpt_v2.calibrate")
    logger.info(
        "Starting calibration fit: method=%s, baseline_run=%s, embargo_cutoff=%s, window_start=%s, rows=%s",
        calibration_config.method,
        baseline_run_id,
        embargo_cutoff.isoformat(),
        calibration_window_start.isoformat() if calibration_window_start else "None",
        rows_with_targets,
    )
    if dropped_missing_targets:
        logger.warning(
            "Dropped %s rows without y_true inside calibration window.", dropped_missing_targets
        )

    calibrator = ForecastCalibrator(calibration_config)
    if calibration_config.method == "none":
        logger.info("Calibration method set to 'none'; skipping model fitting.")
    else:
        calibrator.fit(forecasts_subset, actuals_subset)

    calibrator.save()

    stats = calibrator.get_calibration_stats()
    if not stats.empty:
        logger.info("Calibration models fitted for %d symbol-quantile combinations", len(stats))
        for _, row in stats.iterrows():
            logger.info(
                "Symbol %s quantile %s: method=%s, slope=%.3f, intercept=%.6f, n_samples=%d",
                row["symbol"],
                row["quantile"],
                row["method"],
                row.get("slope", float("nan")),
                row.get("intercept", float("nan")),
                int(row.get("n_samples", 0)) if pd.notna(row.get("n_samples", np.nan)) else 0,
            )
    else:
        logger.warning(
            "No calibration models were fitted. Check min_samples and data availability."
        )

    meta_path = run_dir / "meta.json"
    meta: dict[str, object] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("steps", {})
    meta["command"] = "calibrate"

    calibration_window_start_str = (
        calibration_window_start.isoformat() if calibration_window_start else None
    )
    meta["steps"]["calibrate"] = {
        "run_id": run_id,
        "baseline_run": baseline_run_id,
        "embargo_days": embargo_days,
        "embargo_cutoff": embargo_cutoff.isoformat(),
        "calibration_window_start": calibration_window_start_str,
        "calibration_window_end": embargo_cutoff.isoformat(),
        "calibration_method": calibration_config.method,
        "calibration_window_days": calibration_config.calibration_window_days,
        "rows_after_window": rows_in_window,
        "rows_used": rows_with_targets,
        "models_fitted": int(len(stats)),
        "symbols_trained": sorted(stats["symbol"].unique()) if not stats.empty else [],
        "model_path": calibration_config.model_path,
        "conformal_fallback": calibration_config.conformal_fallback,
        "pit_deviation_threshold": calibration_config.pit_deviation_threshold,
        "conformal_window": calibration_config.conformal_window,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    typer.echo(f"Calibration models saved to {calibration_config.model_path}")


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
