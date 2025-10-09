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

## Accessing GCS Data (local mount)

For the pilot run we mapped the production bucket `gs://jwss_data_store` into the workspace via the
local mount `~/gcs-mount/`. The `check-data` command reads the bronze layer by default using:

- Bucket path: `~/gcs-mount/bronze`
- Template: `stocks/1m/{ticker}/{yyyy}/{ticker}_{yyyy-mm}.parquet`

If your mount or tier differs, adjust `configs/data.yaml` accordingly. The loader uses column aliasing
so raw parquet schemas with `t/o/h/l/c/v` are normalized automatically; only real-time (`session=regular`)
bars between 09:30–16:00 ET are kept.
