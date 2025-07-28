#  HPO in SFP Research Project

This project investigates the impact of hyperparameter optimization (HPO) methods including grid search, Bayesian optimization, and metaheuristics for software fault prediction problem.

## Project Structure
```
project/
├── config/ # Configuration files
├── data/ # Data storage (raw, interim, processed)
├── experiments/ # Experiment scripts and designs
├── notebooks/ # Jupyter notebooks
├── results/ # Results and analysis outputs
├── src/ # Source code
├── tests/ # Test suite
└── docs/ # Documentation
```
## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`

## Usage

1. Configure your experiments in `config/experiments/`
2. Run experiments using scripts in `experiments/scripts/`

## Statistical Testing

This project includes comprehensive statistical testing for comparing HPO methods:

- Parametric and non-parametric tests
- Multiple comparison corrections
- Effect size calculations
- Power analysis

## Contributing

See `CONTRIBUTING.md` for guidelines.
