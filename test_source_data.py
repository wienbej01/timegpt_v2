import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

ET_ZONE = ZoneInfo("America/New_York")

def normalize_like_gcs(df, ticker="AAPL"):
    """Mimic the GCS reader's _normalise_dataframe method."""
    if df.empty:
        return df

    # Handle timestamp column creation
    if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["timestamp"] = df["ts"]
        cols_to_drop = ["ts"]
        if "t" in df.columns:
            cols_to_drop.append("t")
        df = df.drop(columns=cols_to_drop)

    # Rename columns
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vw",
        "n": "n",
        "session": "session",
        "date_et": "date_et"
    }
    df = df.rename(columns=rename_map)

    # Add symbol
    df["symbol"] = ticker

    # Convert timestamp
    timestamps = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        if hasattr(timestamps, 'dt') and timestamps.dt.tz is not None:
            df["timestamp"] = timestamps.dt.tz_convert(ET_ZONE)
        else:
            df["timestamp"] = timestamps.dt.tz_localize(ET_ZONE)

    # Filter to RTH (9:30-16:00 ET)
    timestamps_et = df["timestamp"].dt.tz_convert(ET_ZONE)
    start_time = pd.Timestamp("09:30", tz=ET_ZONE).time()
    end_time = pd.Timestamp("16:00", tz=ET_ZONE).time()
    mask = (timestamps_et.dt.time >= start_time) & (timestamps_et.dt.time < end_time)
    df = df.loc[mask].copy()

    # Sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def main():
    # Read the source Parquet file
    file_path = "/home/jacobw/gcs-mount/bronze/stocks/1m/AAPL/2024/AAPL_2024-08.parquet"
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print("Columns:", list(df.columns))
        print("Data types:")
        print(df.dtypes)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nBasic info:")
        print(df.info())

        # Check for nulls
        nulls = df.isnull().sum()
        print(f"\nNull counts:\n{nulls}")

        # Check for duplicates on timestamp (raw data has 'ts')
        dupes_raw = df.duplicated(subset=['ts']).sum()
        print(f"\nRaw data duplicates on ['ts']: {dupes_raw}")

        if dupes_raw > 0:
            print("Sample raw duplicates:")
            dup_mask_raw = df.duplicated(subset=['ts'], keep=False)
            dup_examples_raw = df[dup_mask_raw].sort_values(['ts']).head(10)
            print(dup_examples_raw[['ts', 'o', 'h', 'l', 'c', 'session']])

        # Check if multiple sessions exist for same ET time
        # Convert ts to ET
        ts_et = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(ET_ZONE)
        df_with_et = df.copy()
        df_with_et['ts_et'] = ts_et

        # Check duplicates on ET time
        dupes_et = df_with_et.duplicated(subset=['ts_et']).sum()
        print(f"\nRaw data duplicates on ET time ['ts_et']: {dupes_et}")

        if dupes_et > 0:
            print("Sample ET time duplicates across sessions:")
            dup_mask_et = df_with_et.duplicated(subset=['ts_et'], keep=False)
            dup_examples_et = df_with_et[dup_mask_et].sort_values(['ts_et', 'session']).head(20)
            print(dup_examples_et[['ts', 'ts_et', 'o', 'c', 'session']])

            # Group by ET time and count sessions
            session_counts_per_et = df_with_et.groupby(['ts_et', 'session']).size().unstack(fill_value=0)
            multiple_sessions = session_counts_per_et.sum(axis=1) > 1
            print(f"\nET times with multiple sessions: {multiple_sessions.sum()}")
            print("Sample ET times with multiple sessions:")
            print(session_counts_per_et[multiple_sessions].head(10))

        # Check if there's a 'symbol' column in raw data
        if 'symbol' in df.columns:
            unique_symbols = df['symbol'].unique()
            print(f"\nUnique symbols in raw data: {unique_symbols}")
            symbol_counts = df['symbol'].value_counts()
            print(f"Symbol counts in raw data:\n{symbol_counts}")
        else:
            print("\nNo 'symbol' column in raw data.")

        # Now simulate GCS normalization
        print("\n=== APPLYING GCS NORMALIZATION ===")
        normalized = normalize_like_gcs(df.copy(), "AAPL")
        print(f"After normalization: {len(normalized)} rows")
        print("Columns after normalization:", list(normalized.columns))
        print("First 5 normalized rows:")
        print(normalized.head())

        # Check for duplicates on normalized data
        dupes_normalized = normalized.duplicated(subset=['symbol', 'timestamp']).sum()
        print(f"\nNormalized data duplicates on ['symbol', 'timestamp']: {dupes_normalized}")

        if dupes_normalized > 0:
            print("Sample normalized duplicates:")
            dup_mask_norm = normalized.duplicated(subset=['symbol', 'timestamp'], keep=False)
            dup_examples_norm = normalized[dup_mask_norm].sort_values(['symbol', 'timestamp']).head(10)
            print(dup_examples_norm[['symbol', 'timestamp', 'open', 'close', 'session']])

        # Check session distribution
        if 'session' in normalized.columns:
            session_counts = normalized['session'].value_counts()
            print(f"\nSession distribution in normalized data:\n{session_counts}")

        # Now simulate _prepare_columns from checks.py
        print("\n=== SIMULATING _prepare_columns ===")
        prepared = normalized.copy()

        # Sort and reset index as in _prepare_columns
        prepared.sort_values(["symbol", "timestamp"], inplace=True)
        prepared.reset_index(drop=True, inplace=True)

        # Check duplicates after prepare
        dupes_prepared = prepared.duplicated(subset=['symbol', 'timestamp']).sum()
        print(f"Rows after prepare: {len(prepared)}")
        print(f"Duplicates after _prepare_columns: {dupes_prepared}")

        if dupes_prepared > 0:
            print("Sample after prepare duplicates:")
            dup_mask_prep = prepared.duplicated(subset=['symbol', 'timestamp'], keep=False)
            dup_examples_prep = prepared[dup_mask_prep].sort_values(['symbol', 'timestamp']).head(10)
            print(dup_examples_prep[['symbol', 'timestamp', 'open', 'close', 'session']])

    except Exception as e:
        print(f"Error reading Parquet file: {e}")

if __name__ == "__main__":
    main()
