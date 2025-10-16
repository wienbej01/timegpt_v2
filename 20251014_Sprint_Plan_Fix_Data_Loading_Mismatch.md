
# 🔧 Sprint Plan — Fix “Data Loading Mismatch & Early Stop” (Claude Code)

**Context (from fault report)**  
- **Universe config:** `2025-01-02` → `2025-02-15`  
- **Loader actually read:** `2024-11-25` → `2025-02-14` (includes warmup)  
- **Forecasts generated:** `2024-11-27` → `2024-12-17` (premature stop in Dec 2024)  
- **Expected:** Forecast generation should extend into **Jan–Feb 2025**.

**Goal**  
Permanently ensure the **scheduler/forecast layer generates snapshots** across the full universe window (inclusive of intended period), **never** halting prematurely at month/year boundaries, and that **IS/OOS evaluation** is **decoupled** from scheduling (classification only in backtest by entry date).

> 🔒 Constraint: **Do not assume file paths.** Claude Code must **discover** relevant modules/classes/functions by searching the repo (grep/find) and adapt patches accordingly.

---

## Outcomes (Definition of Done)

1. A **diagnostic CLI** (or flag) prints: planned snapshot dates/times, and *for each skipped snapshot*, a concrete **reason code** (e.g., `GATE_RTH`, `GATE_MIN_OBS`, `BUDGET`, `MAX_SNAPSHOTS_TOTAL`, `FILTER_PHASE`, `CACHE_SKIP`, `PAYLOAD_EMPTY`).  
2. A **unit test** suite verifying date-range handling across **year boundary** (`2024-12-31 → 2025-01-02`) and **inclusive end date** semantics.  
3. Forecast runner **does not** stop prematurely; **coverage report** shows 2025 snapshots were attempted/executed.  
4. **IS/OOS**: Scheduling **ignores** phase/evaluation; **classification** happens **only** at backtest time using **entry timestamp**.  
5. Cached runs **cannot** silently truncate future days due to key collisions or stale termination flags.

---

## Sprint 0 — Repo Discovery & Guardrails

**Objective**: Find the scheduling/forecast orchestration code; set guardrails for logging and tests.

**Claude Code Tasks**
1. **Search** (ripgrep/find) for likely modules: `scheduler`, `iter_snapshots`, `forecast`, `snapshots`, `batcher`, `timegpt_client`, `runner`.
2. Identify the **entrypoint** used by the CLI (e.g., `cli.py forecast`) and the **scheduler** that yields snapshot timestamps.
3. Locate any **IS/OOS/phase** logic used outside backtest (this is the misuse we must remove from the scheduler).
4. Add a **minimal logging utility** (if not present) to emit structured `INFO` (one line per snapshot attempt/skip). Prefer an **enum** for skip reasons.

**Acceptance**
- Claude lists the target files/functions it will modify in later sprints (paths discovered dynamically).

---

## Sprint 1 — Snapshot Plan Introspection (No Behavior Change)

**Objective**: Expose a **dry-run**/`--print-snapshots` mode that prints all **planned** snapshots given universe dates & preset (before any API call).

**Claude Code Tasks**
1. Add a CLI flag (or a new subcommand) to **print snapshot plan** only:
   - For each trading date in `[start, end]` (inclusive), list snapshot times extracted from the active preset (e.g., `"10:00", "14:30"`), after RTH-window filtering and holidays.
2. Include **per-day counters**: `planned`, `rth_masked`, `final` (post-gates).  
3. Emit **one summary**: total planned snapshots and date coverage.

**Acceptance**
- Running dry-run on the fault case shows **planned snapshots** for **Jan–Feb 2025**, confirming the scheduler itself can produce them (or not, if there’s a bug).

**Test**
- Add a unit test with a fake calendar spanning `2024-12-31` to `2025-01-02` with one snapshot time; assert **3** total planned entries (if inclusive end) or **2** if policy is end-exclusive (we will standardize in Sprint 3).

---

## Sprint 2 — Forecast Loop Instrumentation (Behavior-Preserving)

**Objective**: Instrument the forecast executor to emit **drop/fail reasons** so we can pinpoint where December 2024 premature stop occurs.

**Claude Code Tasks**
1. In the **forecast execution loop**, wrap each snapshot iteration with try/finally and increment counters:
   - `COUNT_PLANNED`, `COUNT_SENT`, `COUNT_SKIPPED`, `COUNT_OK`, `COUNT_FAIL`.
2. On skip/fail, emit **reason code**:
   - `GATE_MIN_OBS`: not enough historical minutes (min_obs_subhourly)
   - `GATE_RTH`: timestamp not in active window
   - `FILTER_DATE`: snapshot filtered by date window logic (bug target)
   - `BUDGET`: per-run/day budget reached
   - `MAX_SNAPSHOTS_TOTAL`: global cap reached (bug suspect)
   - `CACHE_SKIP`: cache indicated “done” for this run/date (bug suspect)
   - `PAYLOAD_EMPTY`: no target rows or payload underflow
   - `ERROR_API`: vendor call error
3. At end, emit a **coverage line**:  
   `COVERAGE planned=### sent=### ok=### fail=### skipped=### by_reason={...}`

**Acceptance**
- Fault reproduction prints a coverage line showing **where** Dec 2024 halts (e.g., `MAX_SNAPSHOTS_TOTAL` or `CACHE_SKIP`).

**Test**
- A tiny fake executor that simulates budget/limit should produce a deterministic coverage report.

---

## Sprint 3 — Date-Window Semantics (Fix Inclusive/Exclusive & Boundary Bugs)

**Objective**: Standardize and test date selection **once** and reuse everywhere.

**Claude Code Tasks**
1. Create a utility (e.g., `date_window.py` or similar) with a **single function**:
   ```python
   def make_trading_day_range(start_date: date, end_date: date, *, inclusive_end: bool = True, tz: str = "America/New_York") -> list[date]
   ```
   - Applies holiday calendar & weekends.
   - **Inclusive end** by default (configurable).
2. Replace any ad-hoc date filters in the scheduler/forecast with this utility.
3. Add **unit tests** for:
   - Year boundary: `2024-12-31` → `2025-01-02` (expect 3 trading dates with inclusive end, skipping weekend/holidays if applicable).
   - End-exclusive toggle: verify 2 dates if `inclusive_end=False`.
   - Off-by-one due to timezone crossing midnight around `snapshot_times`.
4. Emit at startup a **single-line summary**:  
   `DATES start=YYYY-MM-DD end=YYYY-MM-DD inclusive_end=True n_days=## tz=ET`

**Acceptance**
- Dry-run snapshot printing matches expected counts.  
- Codebase uses **only** this utility for date windows.

---

## Sprint 4 — Decouple Evaluation Phases from Scheduling

**Objective**: Ensure **IS/OOS** concepts **never** filter snapshot generation.

**Claude Code Tasks**
1. Grep for `in_sample`, `out_of_sample`, `phase`, `is/oos` usage outside the backtest metrics layer.  
2. If found in **scheduler/forecast**, remove/disable; instead, attach a **pure metadata tag** to snapshots/trades that will be set **later** based on **entry timestamp** only.
3. In backtest aggregation/classification, ensure phase assignment uses **entry date** and a central phase config object (months/ranges).

**Acceptance**
- Unit test: Provide a date that is OOS by config, ensure the scheduler still **plans** it; backtest classification marks trades correctly as OOS (no influence on planning).

---

## Sprint 5 — Limits & Budgets (No Hidden Global Caps)

**Objective**: Confirm caps aren’t silently truncating runs (root cause candidate).

**Claude Code Tasks**
1. Discover and document parameters:
   - `max_snapshots_per_day`, `max_snapshots_total`, `api_budget.per_run`, `per_day`, `cooldown_sec`, cache toggles.
2. Add **validation** at startup:
   - If `max_snapshots_total` is set and `< planned`, **warn** loudly or require explicit `--force`.
   - Print expected snapshot count given settings.
3. If `max_snapshots_total` triggered the Dec stop, propose increasing default or gating it behind **smoke-only** presets.

**Acceptance**
- Running the fault config prints a warning if a cap would truncate 2025 days.

**Test**
- Unit test: with `max_snapshots_total=2` and 5 planned snapshots, verify the warning text and that coverage reports `MAX_SNAPSHOTS_TOTAL` as the skip reason for the remainder.

---

## Sprint 6 — Cache & Keying (Prevent Stale Cutoffs)

**Objective**: Ensure cache cannot falsely indicate the run is complete at a month boundary.

**Claude Code Tasks**
1. Locate caching layer (search `cache`, `artifact`, `quantiles.csv`, `payload hash`).
2. **Version** the cache key to include: `run_id`, `universe_start/end`, `preset name`, **calendar month** `(YYYY-MM)`, and **page/partition index** if batching by day.  
3. On re-run, ensure the executor **continues** from the **last successful snapshot** rather than stopping at the last month with data.
4. Add a `--no-cache` flag for forced cold runs.

**Acceptance**
- Re-running after a partial December completion continues into January (does not stop early).

**Test**
- Simulate a partial run by writing a fake last-success marker; verify the next run resumes **after** that marker and covers January dates.

---

## Sprint 7 — End‑to‑End Coverage Test

**Objective**: Lock regression with an integration test that reproduces the boundary scenario.

**Claude Code Tasks**
1. Add a test harness that mocks:
   - A tiny trading calendar spanning `2024-11-25` → `2025-02-15`,
   - Warmup satisfied before Jan 2,
   - Snapshot preset with 1 snapshot/day.
2. Verify the **coverage** line contains at least one **2025** snapshot sent/ok.
3. Ensure no **FILTER_DATE** or **MAX_SNAPSHOTS_TOTAL** triggered after Dec unless set by test explicitly.

**Acceptance**
- Test fails on the old behavior; passes with the fix.

---

## Sprint 8 — Docs & “Doctor” Command

**Objective**: Give operators tooling to debug quickly next time.

**Claude Code Tasks**
1. Add `cli doctor` or `--doctor` flag that prints:
   - Universe dates, inclusive flag, RTH window, snapshot preset, budget caps, calculated planned snapshots, and any warnings.
2. Extend README with:
   - Date-window semantics,
   - IS/OOS decoupling,
   - Coverage and skip reason taxonomy,
   - How to resume from partial runs (`--no-cache`, resume marker).

**Acceptance**
- Operators can diagnose config vs plan vs execution with one command.

---

## Quick Checklist for Claude (execution order)

1. Discover modules & add snapshot dry‑run printer.  
2. Add forecast loop coverage & skip‑reason logging.  
3. Centralize date-range utility (inclusive end) + tests.  
4. Remove any IS/OOS filter from scheduling; keep classification in backtest only.  
5. Validate caps/budgets; add warnings.  
6. Harden cache keys & resume behavior.  
7. Add E2E coverage test across year boundary.  
8. Add “doctor” diagnostics & README updates.

---

When done, Claude should print:  
**“✅ Forecast coverage fixed: scheduler spans requested 2025 dates; no premature stop; phases decoupled; tests and diagnostics added.”**
