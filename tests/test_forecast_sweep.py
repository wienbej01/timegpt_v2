from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from timegpt_v2.forecast.sweep import ForecastGridPoint, ForecastGridSearch, ForecastGridSpec


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded or {})


def test_grid_spec_cross_product(tmp_path: Path) -> None:
    base_cfg = {
        "snapshot_preset": "baseline",
        "horizon_min": 15,
        "quantiles": [0.25, 0.5, 0.75],
        "target": {"mode": "log_return"},
        "calibration": {"method": "none"},
    }
    spec_cfg = {
        "snapshot_presets": ["baseline", "liquidity_profile"],
        "horizons": [10, 30],
        "quantile_sets": [[0.1, 0.5, 0.9]],
        "target_modes": ["log_return", "volatility_z"],
        "calibration_methods": ["none", "affine"],
    }

    spec = ForecastGridSpec.from_mapping(spec_cfg, base_config=base_cfg)
    points = list(spec.iter_points())
    assert len(points) == 2 * 2 * 1 * 1 * 2 * 2
    sample = points[0]
    assert isinstance(sample, ForecastGridPoint)
    assert sample.snapshot_preset in {"baseline", "liquidity_profile"}
    assert sample.horizon in {10, 30}
    assert sample.target_mode in {"log_return", "volatility_z"}
    assert sample.calibration_method in {"none", "affine"}


def test_forecast_grid_search_plan_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_config_dir = tmp_path / "config"
    base_config_dir.mkdir()
    (base_config_dir / "forecast.yaml").write_text(
        yaml.safe_dump(
            {
                "snapshot_preset": "baseline",
                "horizon_min": 15,
                "quantiles": [0.25, 0.5, 0.75],
                "target": {"mode": "log_return"},
                "calibration": {"method": "none"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_payload = {
        "snapshot_presets": ["baseline"],
        "horizons": [15],
        "quantile_sets": [[0.1, 0.5, 0.9]],
        "target_modes": ["log_return"],
        "calibration_methods": ["none", "affine"],
    }
    spec = ForecastGridSpec.from_mapping(spec_payload, base_config=_load_yaml(base_config_dir / "forecast.yaml"))

    invoked = {"forecast": 0, "backtest": 0, "evaluate": 0}

    def _noop(**kwargs: Any) -> None:
        invoked["noop"] = invoked.get("noop", 0) + 1

    search = ForecastGridSearch(
        base_config_dir=base_config_dir,
        base_forecast_config=_load_yaml(base_config_dir / "forecast.yaml"),
        grid_spec=spec,
        output_root=tmp_path / "output",
        run_id_prefix="RUN",
        forecast_cmd=_noop,
        backtest_cmd=_noop,
        evaluate_cmd=_noop,
    )
    plan = search.run(execute=False, reuse_baseline_artifacts=False)
    assert len(plan) == 2
    assert sorted(plan["calibration_method"].unique().tolist()) == ["affine", "none"]
    first_config_path = Path(plan.iloc[0]["config_path"])
    assert first_config_path.exists()
    override_payload = _load_yaml(first_config_path)
    assert override_payload["quantiles"] == [0.1, 0.5, 0.9]
    assert invoked.get("noop", 0) == 0


def test_collect_metrics_composite_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "test_run"
    eval_dir = tmp_path / "artifacts" / "runs" / run_id / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "pit_coverage": [0.48, 0.53],
            "rmae": [0.8, 0.9],
        }
    )
    diagnostics.to_csv(eval_dir / "forecast_diagnostics.csv", index=False)
    summary = pd.DataFrame([{"sharpe_ratio": 0.75}])
    summary.to_csv(eval_dir / "bt_summary.csv", index=False)

    search = ForecastGridSearch(
        base_config_dir=tmp_path,
        base_forecast_config={},
        grid_spec=ForecastGridSpec(
            snapshot_presets=("baseline",),
            horizons=(15,),
            quantile_sets=((0.25, 0.5, 0.75),),
            level_sets=(tuple(),),
            target_modes=("log_return",),
            calibration_methods=("none",),
        ),
        output_root=tmp_path / "output",
        run_id_prefix="RUN",
        forecast_cmd=lambda **_: None,
        backtest_cmd=lambda **_: None,
        evaluate_cmd=lambda **_: None,
    )

    monkeypatch.chdir(tmp_path)
    metrics = search._collect_metrics(run_id)
    assert pytest.approx(metrics["median_pit"], rel=1e-6) == 0.505
    assert pytest.approx(metrics["pit_deviation"], rel=1e-6) == 0.005
    assert pytest.approx(metrics["median_rmae"], rel=1e-6) == 0.85
    assert pytest.approx(metrics["sharpe_ratio"], rel=1e-6) == 0.75
    assert pytest.approx(metrics["composite_score"], rel=1e-6) == 0.75 * (1 - 0.005)


def test_forecast_grid_scoreboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_config_dir = tmp_path / "config"
    base_config_dir.mkdir()
    forecast_payload = {
        "snapshot_preset": "baseline",
        "horizon_min": 15,
        "quantiles": [0.25, 0.5, 0.75],
        "target": {"mode": "log_return"},
        "calibration": {"method": "none"},
    }
    (base_config_dir / "forecast.yaml").write_text(
        yaml.safe_dump(forecast_payload, sort_keys=False),
        encoding="utf-8",
    )
    spec_payload = {
        "snapshot_presets": ["baseline"],
        "horizons": [15],
        "quantile_sets": [[0.25, 0.5, 0.75]],
        "target_modes": ["log_return"],
        "calibration_methods": ["none", "affine"],
    }
    spec = ForecastGridSpec.from_mapping(spec_payload, base_config=forecast_payload)

    call_index = {"value": 0}

    def _noop(**kwargs: Any) -> None:  # noqa: ARG001
        return

    def _write_eval(**kwargs: Any) -> None:
        call_index["value"] += 1
        run_dir = Path("artifacts") / "runs" / str(kwargs["run_id"])
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        pit = 0.48 + 0.02 * call_index["value"]
        diagnostics = pd.DataFrame(
            {"symbol": ["SYN"], "pit_coverage": [pit], "rmae": [0.8 + 0.05 * call_index["value"]]}
        )
        diagnostics.to_csv(eval_dir / "forecast_diagnostics.csv", index=False)
        summary = pd.DataFrame([{"sharpe_ratio": 0.6 + 0.1 * call_index["value"]}])
        summary.to_csv(eval_dir / "bt_summary.csv", index=False)

    search = ForecastGridSearch(
        base_config_dir=base_config_dir,
        base_forecast_config=forecast_payload,
        grid_spec=spec,
        output_root=tmp_path / "output",
        run_id_prefix="RUN",
        forecast_cmd=_noop,
        backtest_cmd=_noop,
        evaluate_cmd=_write_eval,
    )

    monkeypatch.chdir(tmp_path)
    plan = search.run(execute=True, reuse_baseline_artifacts=False)
    scoreboard_path = tmp_path / "output" / "scoreboard.csv"
    assert scoreboard_path.exists()

    scoreboard = pd.read_csv(scoreboard_path)
    assert "composite_score" in scoreboard.columns
    assert list(scoreboard["run_id"]) == scoreboard.sort_values("composite_score", ascending=False)["run_id"].tolist()

    assert "rank" in plan.columns
    ranks = plan.dropna(subset=["rank"]).sort_values("rank")["rank"].tolist()
    assert ranks == sorted(ranks)