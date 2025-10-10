from datetime import date

import pandas as pd

from timegpt_v2.io.gcs_reader import GCSReader, ReaderConfig
from timegpt_v2.quality.checks import DataQualityChecker

# Config from data.yaml
CONFIG = ReaderConfig(
    bucket="/home/jacobw/gcs-mount/bronze",
    template="stocks/1m/{ticker}/{yyyy}/{ticker}_{yyyy_dash_mm}.parquet",
    skip_timestamp_normalization=True
)

def log_audit(stage, df, description=""):
    rows = len(df) if df is not None else 0
    if df is not None:
        if "symbol" in df.columns and "timestamp" in df.columns:
            dupes = df.duplicated(subset=["symbol", "timestamp"]).sum()
        else:
            dupes = "N/A"
        symbols = df['symbol'].unique().tolist() if "symbol" in df.columns else "N/A"
        print(f"[{stage}] {rows} rows, duplicates: {dupes}, symbols: {symbols} - {description}")
    else:
        print(f"[{stage}] None - {description}")

def forensic_read_universe(tickers, start_date, end_date):
    reader = GCSReader(CONFIG)
    print(f"\n=== FORENSIC AUDIT: Calling read_universe({tickers}, {start_date}, {end_date}) ===")
    all_frames = []

    for ticker in tickers:
        print(f"\n[TICKER {ticker}] Starting read_range")
        ticker_df = reader.read_range(ticker, start_date, end_date)
        log_audit(f"TICKER_{ticker}_FINAL", ticker_df, "ticker data loaded")
        if not ticker_df.empty:
            all_frames.append(ticker_df)

    print(f"\n[CONCAT] Combining {len(all_frames)} ticker frames")
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    duplicates = combined.duplicated(subset=['symbol', 'timestamp']).sum()
    log_audit("CONCAT_FINAL", combined, f"combined, duplicates: {duplicates}")

    return combined

def forensic_quality_check(df):
    print("\n=== QUALITY CHECK FORENSIC ===")
    log_audit("QC_INPUT", df, "input to quality checker")

    checker = DataQualityChecker()
    cleaned, report = checker.validate(df)

    log_audit("QC_OUTPUT", cleaned, "after quality checks")
    print(f"Report passed: {report.passed}")
    for check in report.checks:
        if check.name == "monotonic":
            print(f"Check {check.name}: {check.passed}, details: {check.details}")

    return cleaned, report

def main():
    # Test parameters that might reproduce the large dataset
    # Start with single year, then expand if needed
    start = date(2024, 1, 1)
    end = date(2024, 12, 31)

    print(f"Testing with date range: {start} to {end} AFTER FIX")
    df = forensic_read_universe(["AAPL"], start, end)
    if not df.empty:
        forensic_quality_check(df)
    else:
        print("No data loaded.")

if __name__ == "__main__":
    main()
