# TimeGPT Intraday v2

## Quickstart (≈5 minutes)

1. **Install toolchain** (creates the `.venv`, installs dependencies, and registers pre-commit hooks):

   ```bash
   make install
   ```

2. **Validate the codebase** before making changes:

   ```bash
   make lint
   make test
   ```

3. **Run the end-to-end demo pipeline** on the bundled configs:

   ```bash
   RUN_ID=dev
   python -m timegpt_v2.cli check-data --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli build-features --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli forecast --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli backtest --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli evaluate --config-dir configs --run-id "$RUN_ID"
   python -m timegpt_v2.cli report --config-dir configs --run-id "$RUN_ID"
   ```

   Generated artifacts (features, forecasts, trades, metrics, reports) land under
   `artifacts/runs/<run_id>/...` for inspection.

4. **Explore parameter sweeps** using the trading grid:

   ```bash
   python -m timegpt_v2.cli sweep --config-dir configs --run-id "$RUN_ID"
   ```

- Configuration files live in `configs/` and are the single source of truth for universe definitions,
  scheduler windows, trading rules, and backtest aggregation settings.
- `docs/` contains deeper design notes on data quality, feature engineering, forecasting, trading, and evaluation.
- The `Makefile` mirrors the automation used in CI; running `make fmt && make lint && make test`
  before commits keeps the project reproducible.
