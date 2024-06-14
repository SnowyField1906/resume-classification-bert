# Usage

## Requirements

- `python` and `pip` are required to run the program.
- `miniconda` is recommended to manage the environment.

## Environment setting up

```bash
# Create new Conda environment (optional)
conda create -n resume-classification-bert python=3.9
conda activate resume-classification-bert

# Install dependencies
pip install -r requirements.txt
```

## Linting and formatting

```bash
python -m black .
```