.PHONY: install lint fmt test test-cov run report sweep forecast-grid forecast-grid-plan

install: ## install package + dev
	pip install -e .[dev]

lint:
	ruff check .
	black --check .
	isort --check-only .
	mypy src/timegpt_v2

fmt:
	black .
	isort .

test:
	pytest -q

test-cov:
	pytest --cov=src/timegpt_v2 --cov-report=term-missing

run:
	python -m timegpt_v2.cli run --run-id dev --config-dir configs

report:
	python -m timegpt_v2.cli report --run-id dev --config-dir configs

sweep:
	python -m timegpt_v2.cli sweep --run-id dev --config-dir configs

forecast-grid:
	python -m timegpt_v2.cli sweep \
		--run-id $(RUN_ID) \
		--config-dir configs \
		--forecast-grid configs/forecast_grid.yaml \
		--execute \
		--reuse-baseline \
		--baseline-run $(RUN_ID)

forecast-grid-plan:
	python -m timegpt_v2.cli sweep \
		--run-id $(RUN_ID) \
		--config-dir configs \
		--forecast-grid configs/forecast_grid.yaml \
		--plan-only

demo: ## run full demo pipeline
	python -m timegpt_v2.cli check-data --config-dir configs --run-id demo
	python -m timegpt_v2.cli build-features --config-dir configs --run-id demo
	python -m timegpt_v2.cli forecast --config-dir configs --run-id demo --api-mode offline
	python -m timegpt_v2.cli backtest --config-dir configs --run-id demo
	python -m timegpt_v2.cli evaluate --config-dir configs --run-id demo
	python -m timegpt_v2.cli report --config-dir configs --run-id demo
