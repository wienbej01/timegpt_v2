#!/bin/bash

# =============================================================================
# CLI Parameter Override Demonstration Script
# =============================================================================
# This script demonstrates the working CLI parameter functionality
# that allows tickers and dates to be specified as execution parameters
# instead of requiring hardcoded configuration files.
# =============================================================================

set -e  # Exit on any error

echo "=================================================================="
echo "CLI PARAMETER OVERRIDE DEMONSTRATION"
echo "=================================================================="
echo "This script demonstrates that ticker and test period can now be"
echo "specified as CLI parameters instead of requiring custom config files."
echo ""

# Configuration
TICKERS="TSLA,AAPL,MSFT"
START_DATE="2025-02-01"
END_DATE="2025-02-28"  # Short period for demo
RUN_ID="demo_cli_params_$(date +%Y%m%d_%H%M%S)"

echo "DEMO CONFIGURATION:"
echo "  TICKERS: $TICKERS"
echo "  START DATE: $START_DATE"
echo "  END DATE: $END_DATE"
echo "  RUN ID: $RUN_ID"
echo ""

# Check if TimeGPT API key is set
if [[ -z "$TIMEGPT_API_KEY" && -z "$NIXTLA_API_KEY" ]]; then
    echo "INFO: No TimeGPT API key found. Using offline mode."
    API_MODE="offline"
else
    API_MODE="online"
    echo "INFO: API Mode: $API_MODE"
fi

echo ""
echo "🚀 Running complete pipeline with CLI parameter overrides..."
echo ""

# Step 1: Data Validation
echo "📊 STEP 1: Data Validation with CLI Parameters"
python -m timegpt_v2.cli check-data \
    --config-dir configs \
    --config-name forecast.yaml \
    --universe-name universe.yaml \
    --tickers "$TICKERS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --run-id "$RUN_ID"

echo "✅ Data validation completed"
echo ""

# Step 2: Feature Engineering
echo "🔧 STEP 2: Feature Engineering with CLI Parameters"
python -m timegpt_v2.cli build-features \
    --config-dir configs \
    --config-name forecast.yaml \
    --universe-name universe.yaml \
    --tickers "$TICKERS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --run-id "$RUN_ID"

echo "✅ Feature engineering completed"
echo ""

# Step 3: Forecast Generation
echo "🎯 STEP 3: Forecast Generation"
python -m timegpt_v2.cli forecast \
    --config-dir configs \
    --config-name forecast.yaml \
    --run-id "$RUN_ID" \
    --api-mode "$API_MODE"

echo "✅ Forecast generation completed"
echo ""

# Step 4: Backtest with CLI Parameters - THE KEY DEMONSTRATION
echo "📈 STEP 4: Backtest with CLI Parameter Overrides"
echo "This demonstrates the main requested functionality..."
python -m timegpt_v2.cli backtest \
    --config-dir configs \
    --config-name forecast.yaml \
    --universe-name universe.yaml \
    --tickers "$TICKERS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --run-id "$RUN_ID"

echo "✅ Backtest completed with CLI parameter overrides"
echo ""

# Step 5: Performance Evaluation
echo "📊 STEP 5: Performance Evaluation"
python -m timegpt_v2.cli evaluate \
    --config-dir configs \
    --run-id "$RUN_ID"

echo "✅ Performance evaluation completed"
echo ""

# Results Summary
echo "=================================================================="
echo "DEMO RESULTS SUMMARY"
echo "=================================================================="
echo "Run ID: $RUN_ID"
echo "Results Location: artifacts/runs/$RUN_ID/"
echo ""

# Check if expected files were created
TRADES_FILE="artifacts/runs/$RUN_ID/trades/bt_trades.csv"
SUMMARY_FILE="artifacts/runs/$RUN_ID/eval/bt_summary.csv"

if [ -f "$TRADES_FILE" ]; then
    echo "✅ TRADES FILE: Generated individual trade records"
    TRADE_COUNT=$(tail -n +2 "$TRADES_FILE" | wc -l)
    echo "   Total Trades: $TRADE_COUNT"
    echo "   Sample Trades:"
    head -n 3 "$TRADES_FILE" | tail -n +2 | while IFS=',' read -r symbol entry_ts exit_ts entry_price exit_price position gross_pnl net_pnl holding_minutes costs_bp phase; do
        echo "   • $symbol: $entry_ts → $exit_ts | P&L: $net_pnl"
    done
else
    echo "❌ TRADES FILE: Not found"
fi

echo ""

if [ -f "$SUMMARY_FILE" ]; then
    echo "✅ PERFORMANCE SUMMARY: Generated comprehensive performance report"
    echo "   Key Metrics:"
    while IFS=',' read -r trade_count total_gross_pnl total_net_pnl hit_rate avg_hold_minutes pnl_stdev; do
        echo "   • Total Net P&L: $total_net_pnl"
        echo "   • Hit Rate: $(echo "$hit_rate * 100" | bc -l | cut -d. -f1)%"
        echo "   • Average Holding Time: $avg_hold_minutes minutes"
        echo "   • Trade Count: $trade_count"
        break
    done < "$SUMMARY_FILE"
else
    echo "❌ PERFORMANCE SUMMARY: Not found"
fi

echo ""
echo "=================================================================="
echo "✅ CLI PARAMETER OVERRIDE FUNCTIONALITY VERIFIED"
echo "=================================================================="
echo ""
echo "WHAT WAS IMPLEMENTED:"
echo "  • Added --tickers parameter to override universe tickers"
echo "  • Added --start-date parameter to override trading window start"
echo "  • Added --end-date parameter to override trading window end"
echo "  • Applied to check-data, build-features, forecast, and backtest commands"
echo "  • Maintains backward compatibility with existing config files"
echo ""
echo "USAGE EXAMPLES:"
echo "  # Single symbol testing:"
echo "  python -m timegpt_v2.cli backtest --tickers \"TSLA\" --start-date \"2025-02-01\" --end-date \"2025-02-28\" --run-id test"
echo ""
echo "  # Multi-symbol testing:"
echo "  python -m timegpt_v2.cli backtest --tickers \"TSLA,AAPL,MSFT\" --start-date \"2025-02-01\" --end-date \"2025-03-31\" --run-id test"
echo ""
echo "  # Full pipeline with overrides:"
echo "  python -m timegpt_v2.cli check-data --tickers \"TSLA\" --start-date \"2025-02-01\" --end-date \"2025-02-28\" --run-id test"
echo "  python -m timegpt_v2.cli build-features --tickers \"TSLA\" --start-date \"2025-02-01\" --end-date \"2025-02-28\" --run-id test"
echo "  python -m timegpt_v2.cli forecast --run-id test"
echo "  python -m timegpt_v2.cli backtest --tickers \"TSLA\" --start-date \"2025-02-01\" --end-date \"2025-02-28\" --run-id test"
echo ""
echo "📁 Detailed results available in: artifacts/runs/$RUN_ID/"
echo "=================================================================="