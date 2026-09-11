# Customs Fraud Detection — CSE4214 Pattern Recognition Lab

Comparative ML/DL study on Bangladesh customs declarations: KNN vs Random Forest vs FNN vs Transfer-Learning FNN, with statistical significance testing and explainability.

Dataset: [Kaggle — Customs Dataset](https://www.kaggle.com/datasets/mehreenrahman/customs-dataset)

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

- `data/raw/` — original dataset (not modified)
- `data/processed/` — cleaned/engineered data saved from notebook 02 onward
- `notebooks/` — run in numeric order, 01 through 08
- `src/` — shared functions imported by notebooks (preprocessing, models, evaluation)
- `results/` — saved metrics and figures

See `notebooks/` for the pipeline. Full plan in the project docs (shared separately).
