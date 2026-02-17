# ntxter: Neurotrasmitter - A modular toolkit for data engineering and ML operations

ntxter is a modular toolkit for data engineering and ML operations. It provides
adapters, APIs, and pipelines that help you move raw data through analytics,
feature engineering, model evaluation, and reporting in a repeatable way.

## Key capabilities

- **Unified data adapters:** Common loaders, savers, and sanitizers for tabular
	and analytical workloads.
- **Pipeline-ready abstractions:** Shareable utilities for partitioning,
	statistics, metrics, and reporting that mirror production-ready workflows.
- **API surface:** Thin access layer for data, mlops, reports, and statistics so
	consumers can embed ntxter in services or notebooks.
- **Provider model:** Pluggable providers (IO, filters, observability, plots,
	analysis) to extend functionality without rewriting the core.

## Getting started

```bash
git clone git@github.com:<your-org>/ntxter.git
cd ntxter
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e .
```

`uv` is the default package manager for this project. If you prefer another
tool, adapt the commands accordingly. Python 3.12+ is required (see
`pyproject.toml`).

## Running checks & tests

```bash
uv run pytest
```

Add `-k <expr>` to target specific adapters, providers, or API modules.



## Project layout

- `src/ntxter/adapters`: Data layer adapters (load/save, db access, EDA, stats).
- `src/ntxter/api`: Public API entry points grouped by domain.
- `src/ntxter/core`: Core utilities, pipelines, reports, and mlops helpers.
- `src/ntxter/ports`: Service ports so you can supply custom implementations.
- `tests/`: Unit coverage for adapters, core utilities, and APIs.

```
ntxter/
├── pyproject.toml
├── README.md
├── src/
│   └── ntxter/
│       ├── adapters/
│       │   ├── data/
│       │   ├── mlops/
│       │   ├── pipelines/
│       │   ├── reports/
│       │   └── statistics/
│       ├── api/
│       │   ├── data/
│       │   ├── mlops/
│       │   ├── reports/
│       │   └── statistics/
│       ├── core/
│       │   ├── base/
│       │   ├── data/
│       │   ├── mlops/
│       │   ├── pipelines/
│       │   └── reports/
│       └── ports/
│           ├── data/
│           └── statistics/
└── tests/
	└── unit/
		├── adapters/
		└── core/
```

## Development tips

- Keep adapters small and composable; prefer pure functions for easier testing.
- Document new providers under `src/ntxter/providers` and register them via the
	appropriate registry module.
- When adding pipelines, expose orchestrators through `ntxter.api` so they can
	be reused in batch jobs or services.

## Support

Open an issue with reproduction steps or reach out to the maintainers via your
team channel. Contributions are welcome; please include tests for new features.
