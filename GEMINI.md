# TimeGPT Intraday v2

This document provides a comprehensive overview of the TimeGPT Intraday v2 project, its architecture, and development conventions.

## Project Overview

TimeGPT Intraday v2 is a Python-based platform for time series forecasting, specifically designed for intraday financial data. It provides a complete pipeline for fetching, validating, and forecasting time series data, as well as for backtesting trading strategies based on the forecasts.

The project is structured as a command-line application using the Typer library. The main entry point is `src/timegpt_v2/cli.py`, which provides commands for the different stages of the forecasting pipeline.

### Core Technologies

*   **Python:** The core language of the project.
*   **Typer:** Used for creating the command-line interface.
*   **pandas:** Used for data manipulation and analysis.
*   **pydantic:** Used for data validation and defining data contracts.
*   **scikit-learn:** Likely used for feature engineering and modeling (inferred from the context).
*   **Pytest:** Used for unit testing.
*   **Ruff, Black, isort, mypy:** Used for code linting and formatting.

### Architecture

The project follows a modular architecture, with different modules responsible for specific parts of the forecasting pipeline:

*   **`io`:** Handles data ingestion from sources like Google Cloud Storage (GCS).
*   **`quality`:** Defines data quality contracts and performs data validation.
*   **`fe`:** Contains logic for feature engineering.
*   **`framing`:** Prepares data for the forecasting model.
*   **`forecast`:** Interacts with the TimeGPT forecasting service.
*   **`trading`:** Implements trading rules and logic.
*   **`backtest`:** Provides a simulation engine for backtesting trading strategies.
*   **`eval`:** Contains tools for evaluating the performance of the forecasting model and trading strategies.
*   **`reports`:** Generates reports summarizing the results of a run.
*   **`utils`:** Provides common utility functions.

## Building and Running

The project uses a `Makefile` to simplify the process of building, running, and testing the application.

### Setup

To set up the development environment, run the following command:

```bash
make install
```

This will create a Python virtual environment, install the required dependencies from `pyproject.toml`, and set up pre-commit hooks.

### Running the Application

The main entry point for the application is the `timegpt_v2.cli` module. You can run the application using the `make run` command:

```bash
make run
```

This will execute the `run` command in the CLI, which is a convenience wrapper for running the entire forecasting pipeline.

You can also run individual commands from the CLI. For example, to run the data validation step:

```bash
python -m timegpt_v2.cli check-data --run-id <your_run_id>
```

### Testing

To run the unit tests, use the following command:

```bash
make test
```

To run the linters and code formatters, use the following command:

```bash
make lint
```

## Development Conventions

The project follows a set of development conventions to ensure code quality and consistency.

### Coding Style

*   **Black:** The project uses the Black code formatter to ensure a consistent code style.
*   **isort:** isort is used to automatically sort Python imports.
*   **Ruff:** Ruff is used for linting and identifying potential issues in the code.
*   **mypy:** mypy is used for static type checking.

### Testing

*   All new code should be accompanied by unit tests.
*   Tests are located in the `tests` directory.
*   The project uses pytest for running tests.

### Committing

The project uses pre-commit hooks to automatically run the linters and formatters before each commit. This ensures that all code committed to the repository adheres to the project's coding standards.
