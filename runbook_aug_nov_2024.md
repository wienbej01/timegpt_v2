# Runbook – Aug-Nov 2024 Full-Pipeline Execution

**Date:** 2025-10-10  
**Run ID:** `aug_nov_2024_prod`  
**Universe:** 10 liquid US equities (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, AMD, CRM)  
**Period:** 2024-08-01 → 2024-11-30 (4 months, ~88 trading days)  
**Objective:** Validate Sprint 5 grid sweep on real data; confirm gates; produce immutable artefacts.

---

## 1. Pre-Flight Checklist

| Check | Status | Evidence |
|-------|--------|----------|
| GCS mount reachable | ✅ | `ls ~/gcs-mount/bronze/stocks/1m/` |
| Universe YAML date range correct | ✅ | `configs/universe.yaml` |
| Forecast grid spec present | ✅ | `configs/forecast_grid.yaml` |
| TimeGPT key exported | ✅ | `echo $TIMEGPT_API_KEY | wc -c` > 0 |
| Disk quota ≥ 5 GB free | ✅ | `df -h .` |
| No active runs in `artifacts/` for same ID | ✅ | `! [ -d artifacts/aug_nov_2024_prod ]` |

---

## 2. Execution Commands

```bash
export RUN_ID=aug_nov_2024_prod
set -euo pipefail

# 1. Data validation & feature matrix
python -m timegpt_v2.cli check-data  --config-dir configs --run-id $RUN_ID
python -m timegpt_v2.cli build-features --config-dir configs --run-id $RUN_ID

# 2. Forecast grid sweep (plan first, then execute)
make forecast-grid-plan RUN_ID=$RUN_ID
make forecast-grid RUN_ID=$RUN_ID BASELINE_RUN=$RUN_ID   # reuse own artefacts

# 3. Trading parameter sweep on top-ranked forecast config
python -m timegpt_v2.cli sweep --config-dir configs --run-id $RUN_ID

# 4. Evaluation & report
python -m timegpt_v2.cli evaluate --config-dir configs --run-id $RUN_ID
python -m timegpt_v2.cli report   --config-dir configs --run-id $RUN_ID
```

---

## 3. Expected Outputs

| Artefact | Path | Purpose |
|----------|------|---------|
| Clean data | `artifacts/$RUN_ID/validation/clean.parquet` | DQ gate passed |
| Feature matrix | `artifacts/$RUN_ID/features/features.parquet` | Leakage-safe |
| Forecast grid plan | `artifacts/$RUN_ID/eval/grid/forecast_grid/plan.csv` | All combos enumerated |
| Forecast grid scoreboard | `artifacts/$RUN_ID/eval/grid/forecast_grid/scoreboard.csv` | Ranked by composite score |
| Trading grid summary | `artifacts/$RUN_ID/eval/grid/trading_grid/summary.csv` | k/s sweep results |
| Forecast diagnostics | `artifacts/$RUN_ID/eval/forecast_diagnostics.csv` | Per-symbol rMAE, PIT |
| Portfolio summary | `artifacts/$RUN_ID/eval/portfolio_summary.csv` | OOS Sharpe, hit-rate |
| Robustness report | `artifacts/$RUN_ID/reports/robustness_report.md` | Human-readable summary |

---

## 4. Gate Criteria (must all pass)

| Gate | Threshold | Source File |
|------|-----------|-------------|
| Forecast rMAE | median < 0.95 | `forecast_diagnostics.csv` |
| Forecast rRMSE | median < 0.97 | `forecast_diagnostics.csv` |
| PIT coverage | 48–52 % | `forecast_diagnostics.csv` |
| OOS Sharpe | ≥ 0.5 | `portfolio_summary.csv` |
| OOS net P&L | > 0 | `portfolio_summary.csv` |
| Hit-rate | ≥ 48 % | `portfolio_summary.csv` |
| Cost 1.5× | net P&L ≥ 0 | `cost_sensitivity.csv` |

If any gate fails, `evaluate` exits non-zero and CI aborts.

---

## 5. Monitoring & Alerts

- **Data gaps**: `check-data` logs rows_before / rows_after; delta > 5 % → warning.
- **API latency**: `forecast` logs per-call latency; > 30 s → retry with backoff.
- **Cache hit rate**: printed in `forecast.log`; < 50 % on rerun → investigate.
- **Slippage drift**: `backtest` logs avg slippage vs config; > 2× config → flag.
- **Kill switch**: if `evaluate` exits non-zero, pipeline stops; artefacts still immutable.

---

## 6. Immutable Run Artefacts

All files under `artifacts/$RUN_ID/` are write-once. A second invocation with same ID aborts unless `--force` passed. MD5 checksums of key CSVs are appended to `meta.json` for audit.

---

## 7. Post-Run Review

1. Inspect `robustness_report.md` – confirm narrative matches numbers.
2. Open `scoreboard.csv` – top combo should satisfy gating thresholds.
3. Archive `meta.json` and checksums to long-term storage.
4. Update `SYSTEM_TECH_DOC.md` with run date, commit hash, and any anomalies.
5. Schedule next run (weekly automation) via cron referencing this runbook.

---

End of Runbook