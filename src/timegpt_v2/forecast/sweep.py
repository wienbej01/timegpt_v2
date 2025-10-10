from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import typer
import yaml

GridCommand = Callable[..., None]


@dataclass(frozen=True)
class ForecastGridPoint:
    """Concrete forecast configuration under evaluation."""

    snapshot_preset: str
    horizon: int
    quantiles: tuple[float, ...]
    levels: tuple[int, ...]
    target_mode: str
    calibration_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_preset": self.snapshot_preset,
            "horizon": self.horizon,
            "quantiles": list(self.quantiles),
            "levels": list(self.levels),
            "target_mode": self.target_mode,
            "calibration_method": self.calibration_method,
        }

    def combo_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return md5(payload, usedforsecurity=False).hexdigest()  # noqa: S324


@dataclass(frozen=True)
class ForecastGridSpec:
    """Cross-product specification for forecast configuration sweeps."""

    snapshot_presets: tuple[str, ...]
    horizons: tuple[int, ...]
    quantile_sets: tuple[tuple[float, ...], ...]
    level_sets: tuple[tuple[int, ...], ...]
    target_modes: tuple[str, ...]
    calibration_methods: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_config: Mapping[str, Any]) -> ForecastGridSpec:
        def _ensure_tuple(sequence: Iterable[Any]) -> tuple[Any, ...]:
            return tuple(sequence)

        def _extract(key: str, default: Sequence[Any]) -> tuple[Any, ...]:
            raw = payload.get(key)
            if raw is None:
                return tuple(default)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                return tuple(raw)
            raise typer.BadParameter(f"forecast grid option '{key}' must be a sequence")

        snapshot_presets = _extract(
            "snapshot_presets",
            [str(base_config.get("snapshot_preset", ""))] if base_config.get("snapshot_preset") else [],
        )
        if not snapshot_presets:
            raise typer.BadParameter("forecast grid must define at least one snapshot preset")

        horizons = _extract("horizons", [int(base_config.get("horizon_min", 15))])
        quantile_sets = tuple(
            tuple(float(q) for q in seq)
            for seq in _extract("quantile_sets", [base_config.get("quantiles", [0.25, 0.5, 0.75])])
        )
        level_sets = tuple(
            tuple(int(level) for level in seq) for seq in _extract("level_sets", [base_config.get("levels", [])])
        )
        if not level_sets:
            level_sets = (tuple(),)
        target_modes = tuple(
            str(mode)
            for mode in _extract(
                "target_modes",
                [str(base_config.get("target", {}).get("mode", "log_return"))],
            )
        )
        calibration_methods = tuple(
            str(method)
            for method in _extract(
                "calibration_methods",
                [str(base_config.get("calibration", {}).get("method", "none"))],
            )
        )

        return cls(
            snapshot_presets=snapshot_presets,
            horizons=tuple(int(h) for h in horizons),
            quantile_sets=quantile_sets,
            level_sets=level_sets,
            target_modes=target_modes,
            calibration_methods=calibration_methods,
        )

    def iter_points(self) -> Iterable[ForecastGridPoint]:
        for preset in self.snapshot_presets:
            for horizon in self.horizons:
                for quantiles in self.quantile_sets:
                    for levels in self.level_sets:
                        for target_mode in self.target_modes:
                            for calibration_method in self.calibration_methods:
                                yield ForecastGridPoint(
                                    snapshot_preset=str(preset),
                                    horizon=int(horizon),
                                    quantiles=tuple(float(q) for q in quantiles),
                                    levels=tuple(int(level) for level in levels),
                                    target_mode=str(target_mode),
                                    calibration_method=str(calibration_method),
                                )


class ForecastGridSearch:
    """Coordinate multi-configuration forecasting sweeps."""

    def __init__(
        self,
        *,
        base_config_dir: Path,
        base_forecast_config: Mapping[str, Any],
        grid_spec: ForecastGridSpec,
        output_root: Path,
        run_id_prefix: str,
        forecast_cmd: GridCommand,
        backtest_cmd: GridCommand,
        evaluate_cmd: GridCommand,
        logger: logging.Logger | None = None,
        baseline_run: str | None = None,
    ) -> None:
        self._base_config_dir = base_config_dir
        self._base_forecast_config = copy.deepcopy(base_forecast_config)
        self._grid_spec = grid_spec
        self._output_root = output_root
        self._run_id_prefix = run_id_prefix
        self._forecast_cmd = forecast_cmd
        self._backtest_cmd = backtest_cmd
        self._evaluate_cmd = evaluate_cmd
        self._logger = logger or logging.getLogger(__name__)
        self._baseline_run = baseline_run

    def run(self, *, execute: bool, reuse_baseline_artifacts: bool) -> pd.DataFrame:
        """Enumerate (and optionally execute) forecast configurations."""
        plan_rows: list[dict[str, Any]] = []
        self._output_root.mkdir(parents=True, exist_ok=True)

        baseline_paths: dict[str, Path] | None = None
        if execute and reuse_baseline_artifacts:
            if self._baseline_run is None:
                raise typer.BadParameter("reuse_baseline_artifacts requires --baseline-run")
            baseline_paths = self._resolve_baseline_artifacts(self._baseline_run)

        for index, point in enumerate(self._grid_spec.iter_points(), start=1):
            combo_hash = point.combo_hash()
            combo_dir = self._output_root / combo_hash
            config_dir = combo_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            self._materialize_config_dir(config_dir)
            forecast_config_path = config_dir / "forecast.yaml"
            self._write_override_config(point, forecast_config_path)

            combo_run_id = f"{self._run_id_prefix}_{index:03d}_{combo_hash[:6]}"
            metrics: dict[str, Any] | None = None
            if execute:
                self._logger.info(
                    "forecast_grid.execute",
                    extra={"combo_hash": combo_hash, "run_id": combo_run_id, **point.to_dict()},
                )
                metrics = self._execute_combo(
                    run_id=combo_run_id,
                    config_dir=config_dir,
                    baseline_artifacts=baseline_paths,
                )
            else:
                self._logger.info(
                    "forecast_grid.plan",
                    extra={"combo_hash": combo_hash, "run_id": combo_run_id, **point.to_dict()},
                )

            row: dict[str, Any] = {
                "order": index,
                "run_id": combo_run_id,
                "combo_hash": combo_hash,
                "snapshot_preset": point.snapshot_preset,
                "horizon": point.horizon,
                "quantiles": ";".join(f"{q:.3f}" for q in point.quantiles),
                "levels": ";".join(str(level) for level in point.levels),
                "target_mode": point.target_mode,
                "calibration_method": point.calibration_method,
                "config_path": str(forecast_config_path),
            }
            if metrics is not None:
                row.update(metrics)
            plan_rows.append(row)

        plan_df = pd.DataFrame(plan_rows)
        if plan_df.empty:
            plan_df.to_csv(self._output_root / "plan.csv", index=False)
            return plan_df

        if "run_id" not in plan_df.columns:
            raise ValueError("Forecast grid plan is missing run identifiers")

        plan_df.sort_values("run_id", inplace=True)

        scoreboard_path = self._output_root / "scoreboard.csv"
        if "composite_score" in plan_df.columns:
            scored = plan_df.dropna(subset=["composite_score"]).copy()
            if not scored.empty:
                scored.sort_values(
                    by=["composite_score", "median_rmae", "pit_deviation", "run_id"],
                    ascending=[False, True, True, True],
                    inplace=True,
                )
                scored["rank"] = range(1, len(scored) + 1)
                plan_df = plan_df.merge(scored[["run_id", "rank"]], on="run_id", how="left")
                scored.to_csv(scoreboard_path, index=False)
            elif scoreboard_path.exists():
                scoreboard_path.unlink(missing_ok=True)
        elif scoreboard_path.exists():
            scoreboard_path.unlink(missing_ok=True)

        if "rank" in plan_df.columns:
            plan_df.sort_values(
                by=["rank", "run_id"],
                inplace=True,
                na_position="last",
            )
        plan_path = self._output_root / "plan.csv"
        plan_df.to_csv(plan_path, index=False)
        return plan_df

    def _materialize_config_dir(self, destination: Path) -> None:
        for yaml_path in self._base_config_dir.glob("*.yaml"):
            shutil.copy2(yaml_path, destination / yaml_path.name)

    def _write_override_config(self, point: ForecastGridPoint, path: Path) -> None:
        config = copy.deepcopy(self._base_forecast_config)
        config["snapshot_preset"] = point.snapshot_preset
        config["horizon_min"] = point.horizon
        config["quantiles"] = list(point.quantiles)
        if point.levels:
            config["levels"] = list(point.levels)
        elif "levels" in config:
            del config["levels"]

        target_cfg = config.setdefault("target", {})
        target_cfg["mode"] = point.target_mode

        calibration_cfg = config.setdefault("calibration", {})
        calibration_cfg["method"] = point.calibration_method

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def _resolve_baseline_artifacts(self, baseline_run: str) -> dict[str, Path]:
        base_run_dir = Path("artifacts") / "runs" / baseline_run
        validation_path = base_run_dir / "validation" / "clean.parquet"
        features_path = base_run_dir / "features" / "features.parquet"

        if not validation_path.exists():
            raise FileNotFoundError(
                f"Baseline run '{baseline_run}' is missing validation data at {validation_path}"
            )
        if not features_path.exists():
            raise FileNotFoundError(
                f"Baseline run '{baseline_run}' is missing feature matrix at {features_path}"
            )
        return {"validation": validation_path, "features": features_path}

    def _execute_combo(
        self,
        *,
        run_id: str,
        config_dir: Path,
        baseline_artifacts: dict[str, Path] | None,
    ) -> dict[str, Any]:
        run_dir = Path("artifacts") / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if baseline_artifacts is not None:
            validation_dest = run_dir / "validation"
            validation_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_artifacts["validation"], validation_dest / "clean.parquet")
            features_dest = run_dir / "features"
            features_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_artifacts["features"], features_dest / "features.parquet")

        self._invoke(
            self._forecast_cmd,
            config_dir=config_dir,
            run_id=run_id,
        )
        self._invoke(
            self._backtest_cmd,
            config_dir=config_dir,
            run_id=run_id,
        )
        self._invoke(
            self._evaluate_cmd,
            config_dir=config_dir,
            run_id=run_id,
        )
        return self._collect_metrics(run_id)

    def _collect_metrics(self, run_id: str) -> dict[str, Any]:
        eval_dir = Path("artifacts") / "runs" / run_id / "eval"
        diagnostics_path = eval_dir / "forecast_diagnostics.csv"
        summary_path = eval_dir / "bt_summary.csv"

        if not diagnostics_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Expected evaluation outputs missing for run {run_id}: "
                f"{diagnostics_path} or {summary_path}"
            )

        diagnostics = pd.read_csv(diagnostics_path)
        if diagnostics.empty:
            raise ValueError(f"forecast diagnostics empty for run {run_id}")

        median_pit = float(diagnostics["pit_coverage"].median())
        median_rmae = float(diagnostics["rmae"].median())
        pit_deviation = abs(0.5 - median_pit)

        summary = pd.read_csv(summary_path)
        sharpe = float(summary.iloc[0].get("sharpe_ratio", np.nan))

        composite_score = float(np.nan)
        if not np.isnan(sharpe):
            composite_score = sharpe * max(0.0, 1.0 - pit_deviation)

        return {
            "median_pit": median_pit,
            "pit_deviation": pit_deviation,
            "median_rmae": median_rmae,
            "sharpe_ratio": sharpe,
            "composite_score": composite_score,
        }

    @staticmethod
    def _invoke(command: GridCommand, **kwargs: Any) -> None:
        try:
            command(**kwargs)
        except typer.Exit as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Command {command.__name__} exited with code {exc.exit_code}") from exc