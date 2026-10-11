# Development

Install with `pip install -e '.[dev]'` and run `pytest`.
Targeted linting can be run with `ruff check --select E9,F63,F7,F82 src tests scripts`.
Use numerical regression checks for changes to densities, losses, sampling,
distributed seeding, or evaluation. Keep the NFMD reproduction entry points and
baseline recipes distinct. Document changes to reference settings in `docs/bgs.md`.
