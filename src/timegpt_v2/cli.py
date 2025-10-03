from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="TimeGPT Intraday v2 command line interface")


def _default_config_dir() -> Path:
    return Path("configs")


@app.command(name="check-data")
def check_data(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Validate source data before downstream processing."""
    typer.echo(f"check-data stub: run_id={run_id}, config_dir={config_dir}")


@app.command(name="build-features")
def build_features(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Build feature matrix from validated data."""
    typer.echo(f"build-features stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def forecast(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Generate TimeGPT quantile forecasts."""
    typer.echo(f"forecast stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def backtest(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Run trading backtest using generated forecasts."""
    typer.echo(f"backtest stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def evaluate(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Evaluate model and trading performance metrics."""
    typer.echo(f"evaluate stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def report(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Assemble final report for the run."""
    typer.echo(f"report stub: run_id={run_id}, config_dir={config_dir}")


@app.command()
def sweep(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
    grid_config: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Execute trading parameter sweep."""
    typer.echo(
        f"sweep stub: run_id={run_id}, config_dir={config_dir}, grid={grid_config}"
    )


@app.command()
def run(
    config_dir: Path = typer.Option(default=_default_config_dir(), exists=True, file_okay=False),
    run_id: str = typer.Option(..., help="Unique run identifier"),
) -> None:
    """Convenience pipeline runner (scaffold placeholder)."""
    typer.echo(f"run stub: run_id={run_id}, config_dir={config_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
