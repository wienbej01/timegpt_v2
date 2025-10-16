
# 2025-10-14 — Sprint Plan to Fix **Hidden Date Backfill / Trading-Window Violations**
**Filename:** `20251014_date_issue_prompt.md`  
**Owner:** Jacob (PM) — Chief Data Scientist (Sponsor)  
**Implementer:** Claude Code (in-repo code editor/agent)  
**Criticality:** 🚨 P0 — Users’ configured trading periods are silently violated

---

## 🧨 Problem Statement (What’s broken)
- In `gcs_loader.py`, the loader **always** sets:
  ```python
  hist_start = start - timedelta(days=rolling_history_days)
  ```
- Result: when the user asks for **Jan 2, 2025 → Feb 15, 2025**, the system loads **Dec 13, 2024 → Feb 15, 2025** and the **forecast/backtest start earlier** than the user’s intended trading period.
- Users expect **universe dates** to be the **actual trading window**, but the system **trades** as soon as there’s enough history, **not** when the configured period begins.

**Consequences**
1. Trading happens **outside** requested period (silent early trades).  
2. Universe “dates” do **not** mean “trading window”, creating confusion.  
3. No explicit control: warmup vs trading window are coupled & hidden.  
4. Reproducibility and evaluation slices (IS/OOS) are **contaminated** by unrequested periods.

---

## 🎯 Design Goals (Definition of Done)
1. **Warmup/Trading Decoupling:** Users can independently configure:
   - **`trading_window`**: the **only** period when snapshots/trades can occur.
   - **`history_backfill_days`**: optional **warmup** loaded **before** trading_window for model context.
2. **Hard Clamp:** Forecast snapshots **must not** be emitted before `trading_window.start` or after `trading_window.end` — even if history is loaded earlier/later.
3. **Transparent Defaults:** If `history_backfill_days` > 0, CLI prints a **one-line notice** showing the data load range vs. trading window.
4. **Doctor/Explainability:** A `doctor` output shows **exact ranges** (`load_start/load_end`, `trade_start/trade_end`) and **why any snapshot was skipped**.
5. **Tests:** Unit + E2E tests lock the policy across **year boundaries** & **RTH filters**.
6. **Backwards Compatibility:** Old configs continue to run; default behavior is **no trading outside universe dates**; warmup is **explicit**.

---

## ⚙️ New/Changed Configuration
Add **explicit** knobs (no file paths assumed; Claude must find & patch config parsing):
```yaml
# in configs/universe*.yaml (or dedicated backtest/forecast config section)
trading_window:
  start: "2025-01-02"
  end:   "2025-02-15"
history_backfill_days: 20   # optional historical bars loaded BEFORE trading_window.start
enforce_trading_window: true   # default TRUE: hard-clamp snapshot planning
```

**Notes**
- If legacy configs only have `dates: {start, end}`, treat them as `trading_window` and set `history_backfill_days` from previous `rolling_history_days` **but clamp snapshots** to trading window.
- Keep **feature/loader** warmup separate from **scheduler/trader** clamp.

---

## 📦 Artifacts to Modify (Claude must discover exact paths)
- Loader: `gcs_loader.py` (or equivalent) — **load range** behavior
- Scheduler: `forecast/scheduler.py` (or equivalent) — **snapshot planning** clamp
- Forecast CLI: `cli.py` (forecast subcommand) — **wiring & notices**
- Docs/README — **semantics & examples**
- Tests: `tests/` — unit and integration

> Claude **must search** for these modules and adapt if names differ.

---

## 🗺️ Sprint Breakdown

### **Sprint 0 — Repo Discovery & Guardrails (0.5d)**
**Tasks**
1. Search project for loader/scheduler/CLI modules: keywords: `gcs_loader`, `rolling_history_days`, `iter_snapshots`, `snapshot_preset`, `forecast`, `cli`.
2. Add a tiny logging util (if missing) for one-line **INFO** with structured fields.
3. Document list of candidate files to patch.

**Acceptance**
- Print discovered file paths and chosen patch points.
- ✅ *“Sprint 0 done”*

---

### **Sprint 1 — Introduce Warmup vs Trading Window (1d)**
**Tasks**
1. Introduce new config keys (`trading_window`, `history_backfill_days`, `enforce_trading_window`) with sane defaults and backwards compatibility.
2. In the **loader**, compute **two ranges**:
   - `load_start = trading_window.start - history_backfill_days`
   - `load_end = trading_window.end`
3. Ensure **data load** covers `[load_start, load_end]` (for features/warmup). **Do NOT** change how features are built.
4. Add startup notice:
   - `RANGES load=[YYYY-MM-DD, YYYY-MM-DD] trade=[YYYY-MM-DD, YYYY-MM-DD] backfill_days=##`

**Acceptance**
- Loader reads from widened range; no change to feature logic.
- Notice prints once per run.
- ✅ *“Sprint 1 done”*

**Tests**
- Unit: given `trading_window` & `history_backfill_days`, computed `load_start`/`load_end` are correct (including year boundary).

---

### **Sprint 2 — Clamp Snapshot Planning to Trading Window (1d)**
**Tasks**
1. In the **scheduler**, **clamp** snapshot **dates** to the `trading_window`. Even if data exist earlier, **no snapshot** before `trade_start`.
2. Keep warmup for **min_obs_subhourly** satisfied **by loaded history**, but **disallow** snapshot generation outside trading window.
3. Add a **skip reason** taxonomy (if not present):
   - `SKIP_BEFORE_TRADE_WINDOW`, `SKIP_AFTER_TRADE_WINDOW`, `GATE_MIN_OBS`, `GATE_RTH`, etc.

**Acceptance**
- Dry-run shows **no planned snapshots** before `trade_start`.
- Coverage line after run includes skip reasons only within the trade period.
- ✅ *“Sprint 2 done”*

**Tests**
- Unit: `trading_window.start=2025-01-02`, `history_backfill_days=20` → planned snapshots start on **2025‑01‑02** even though loader pulled Dec data.

---

### **Sprint 3 — Forecast Loop Enforcement & Friendly Errors (0.5d)**
**Tasks**
1. In the **forecast runner**, add a **preflight check** that aborts any snapshot with `ds < trade_start or ds > trade_end` and logs `SKIP_*` reason.
2. If `enforce_trading_window=false`, allow legacy behavior but print a **WARN** banner that trading may occur outside configured dates.

**Acceptance**
- Attempting to forecast outside window **never** calls vendor API.
- WARN banner shows when permissive mode is on.
- ✅ *“Sprint 3 done”*

**Tests**
- Unit: with `enforce_trading_window=true`, confirm no API call for a date one day before `trade_start`.

---

### **Sprint 4 — Backtest Clamp & Phase Semantics (1d)**
**Tasks**
1. Ensure **backtest trade construction** discards any entry/exit **outside** `trading_window` (hard stop). Stage this at blotter creation or pre‑aggregation.
2. Phase (IS/OOS) assignment uses **entry timestamp** only; remove any scheduler gating tied to phases.
3. Add **summary** lines to evaluation: trades_by_phase counts consider **window clamp**.

**Acceptance**
- No backtest trades before `trade_start` or after `trade_end`.
- Phase classification independent of scheduling; based on **entry** only.
- ✅ *“Sprint 4 done”*

**Tests**
- Unit: synthetic signals creating pre‑start entries → result 0 trades.
- Unit: entries on start day → accepted.

---

### **Sprint 5 — “Doctor” Diagnostics & Coverage (0.5d)**
**Tasks**
1. Add `cli doctor` (or `--doctor`) to print:
   - `load_start/load_end`, `trade_start/trade_end`, backfill days
   - planned snapshots vs final
   - any caps (`max_snapshots_total`) and cache resume info
2. At end of forecast, print coverage:
   - `COVERAGE planned=## sent=## ok=## skipped=## by_reason={...}`

**Acceptance**
- Operators can confirm **at a glance** whether trading is constrained to the intended window.
- ✅ *“Sprint 5 done”*

**Tests**
- Unit: `doctor` returns values matching computed load/trade ranges.

---

### **Sprint 6 — Backward Compatibility & Migration (0.5d)**
**Tasks**
1. If legacy `dates: {start, end}` found and `trading_window` absent:
   - Treat `dates` as **trading_window**,
   - Derive `history_backfill_days` from old `rolling_history_days` (if present),
   - **Clamp snapshots** by default.
2. Emit one-time **migration warning** with a link to docs (“new semantics: warmup vs trading window”).

**Acceptance**
- Legacy configs run without surprises; trades occur only inside `dates` unless user disables enforcement explicitly.
- ✅ *“Sprint 6 done”*

**Tests**
- Unit: legacy config emulation → verify clamp + migration warning.

---

### **Sprint 7 — End-to-End Boundary Test (1d)**
**Tasks**
1. Build a tiny E2E harness:
   - `trade_start=2025-01-02`, `trade_end=2025-02-15`, `history_backfill_days=20`,
   - 1 snapshot/day preset, RTH filtering on.
2. Assert: first planned snapshot is **2025‑01‑02**; **no** December snapshots; trades only within window.

**Acceptance**
- Test fails on old behavior; passes post‑fix.
- ✅ *“Sprint 7 done”*

---

## 🧪 Test Matrix (Summary)
- **Date math** across year boundary (Dec → Jan).  
- **RTH** filter interaction with clamp.  
- **Warmup** satisfied by history; **no snapshots** pre‑start.  
- **Strict vs permissive** (`enforce_trading_window`).  
- **Legacy config** migration path.  
- **Coverage/skip reasons** appear as designed.

---

## 📣 UX & Docs
- Update README: “**Warmup vs Trading Window**” with diagrams & examples.
- Add examples:
  - *Example A*: `history_backfill_days=20`, `enforce_trading_window=true` → prints widened load range but trades only in requested dates.
  - *Example B*: permissive mode with WARN banner.

---

## 🔁 Rollback & Risks
- **Edge risk**: users expecting old permissive trading may see fewer trades → provide `enforce_trading_window=false` escape hatch.
- **Cache keys**: ensure cache/resume logic includes **trading_window** so re-runs don’t pick stale prefix snapshots.
- **Holiday calendars**: confirm date clamp works with half-days/holidays; unit tests should stub a simple calendar provider.

---

## ✅ Final Acceptance Criteria
- Snapshots & trades are **strictly limited** to `trading_window` when enforcement is on.
- Data loading may extend backward for warmup, but **no trading** occurs before `trade_start`.
- `doctor` prints all ranges and reasons; coverage includes skip taxonomy.
- Unit + E2E tests pass across year boundary.
- Docs updated; migration warning for legacy users.

When complete, print:
> **“✅ Trading window clamp enforced. Warmup and trading are decoupled; coverage and doctor diagnostics added; tests green.”**
