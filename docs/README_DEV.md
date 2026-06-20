# Development Setup

To run tests from the root directory of the repository, you need to install the `deepagents` package in editable mode in your Python environment.

## Quick Setup

From the root directory, run:

```bash
pip install -e libs/deepagents
```

Or use the setup script:

```bash
./setup_dev.sh
```

## Using uv (Recommended)

If you're using `uv` for dependency management:

```bash
cd libs/deepagents
uv pip install -e .
```

## Using conda

If you're using conda:

```bash
# Activate your conda environment first
conda activate deepagents  # or your environment name

# Then install in editable mode
pip install -e libs/deepagents
```

## Verify Installation

After installation, verify it works:

```bash
python3 -c "from deepagents.middleware.datetime import DateTimeMiddleware; print('✅ Import successful')"
```

## Running Tests

Once installed, you can run tests from the root directory:

```bash
# Run all datetime middleware tests
pytest tests/test_datetime_middleware.py -v

# Run all tests
pytest tests/ -v
```

## Note

The package is already installed in the `libs/deepagents/.venv` environment. If you want to use a different Python environment (like conda), you need to install it there as well.
