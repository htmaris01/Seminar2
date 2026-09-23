# Toxic Comment Classification

Project for **Artificial Intelligence for Software Engineering**.

The application solves the Kaggle Jigsaw Toxic Comment Classification Challenge as a
multi-label classification problem with six labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## Project Structure

```text
data/
  raw/                 # place Kaggle files here
  processed/           # train.csv after cleaning phase
models/                # optional saved models
notebooks/             # notebooks
report/                # final reports
results/               # generated metrics
src/toxic_comments/    # application code
tests/                 # unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m toxic_comments --include-transformers --folds 5
```

Download the dataset from Kaggle and put `train.csv` at:

```text
data/raw/train.csv
```

