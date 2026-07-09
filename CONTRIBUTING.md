# Contributing

Thanks for your interest in improving Fruit Quality Detection.

## Getting started
1. Fork and clone the repository.
2. Create a virtual environment and `pip install -r requirements.txt`.
3. Create a feature branch: `git checkout -b feat/short-description`.

## Development
- Keep the `common/` pipeline framework-agnostic (no TF/PyTorch imports there).
- Add or update tests in `tests/` for any behavior change.
- Run the suite before pushing: `python -m pytest tests/ -q`.

## Commit style
Use Conventional Commits:

- `feat:` new capability
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` non-behavioral code change
- `perf:` performance improvement
- `test:` tests only
- `build:` / `ci:` / `chore:` tooling

One logical change per commit. Avoid vague messages like "update" or "fixes".

## Pull requests
- Describe what changed and why.
- Link related issues.
- Note any impact on the reported metrics or hardware behavior.
- Do not commit datasets, weights, or secrets.
