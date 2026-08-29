# Gripping Parameter Estimation

Tools and experiments for estimating gripping parameters (friction, stiffness, contact geometry, etc.) using Python. This repository collects data-processing code, model implementations, and example notebooks used to fit and evaluate parameter-estimation methods for robotic/physical gripping tasks.

## Features

- Data ingestion and preprocessing pipelines for gripping experiments
- Modular model implementations (Bayesian, optimization-based, ML) in Python
- Jupyter notebooks demonstrating experiments, visualizations, and evaluation
- Tests and utilities for reproducible experiments

## Repository layout

```
README.md           # this file
src/                 # Python package / modules (models, data, utils)
notebooks/           # Jupyter notebooks with experiments and visualizations
data/                # (optional) raw and processed datasets; typically gitignored
tests/               # pytest test suite
scripts/             # helper scripts for running experiments, evaluation, and plotting
requirements.txt     # pinned Python dependencies (or pyproject.toml)
Dockerfile           # optional: container for reproducible runs
```

## Getting started

Recommended: use a virtual environment or Conda environment.

Install dependencies (pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the repository uses Poetry / pyproject.toml:

```bash
poetry install
poetry shell
```

Run the notebooks (recommended via Jupyter Lab):

```bash
pip install jupyterlab
jupyter lab notebooks/
```

Or run a simple experiment script (example):

```bash
python scripts/run_experiment.py --config configs/example.yaml
```

## Tests

Run the test suite with pytest:

```bash
pip install -r requirements-dev.txt  # if present
pytest -q
```

## Contributing

Contributions welcome. Please open issues for bug reports or feature requests and submit PRs for fixes. Add tests for new features and follow the existing code style.

## License

If this project is intended to be open source, add a LICENSE file. Otherwise, keep the license section here as a placeholder.

## Contact

Created by milton-stark. If you have questions, open an issue or contact the repository owner on GitHub.
