# FraudGuard-ML

Machine learning–based credit card fraud detection system for batch transaction analysis.

Fraud detection is a classic imbalanced classification problem: fraudulent transactions are rare, but missing them is costly. This project walks through the full applied ML workflow — exploratory analysis, preprocessing, model training, evaluation, and a simple Streamlit demo — at an intermediate portfolio level.

## Features

- Exploratory data analysis notebook with class, amount, time, and correlation views
- Stratified train / validation / test split (70% / 15% / 15%)
- Class-weighted **Logistic Regression** and **XGBoost** training
- Model selection by validation **Average Precision** (PR-AUC)
- Decision threshold tuned on validation data (maximize F1-Score)
- Final evaluation reported once on a held-out test set
- Streamlit app for CSV upload, fraud probability scoring, and result download

## Technologies

- Python 3.12+
- pandas, NumPy
- scikit-learn, XGBoost
- matplotlib, seaborn
- Streamlit, joblib, Jupyter

## Dataset

**Source:** [Credit Card Fraud Detection (ULB) on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraudulent | 492 (~0.17%) |
| Features | `Time`, `V1`–`V28`, `Amount` |
| Target | `Class` (0 = legitimate, 1 = fraud) |

`V1`–`V28` are PCA-anonymized features. `Time` is elapsed seconds since the first transaction.

**Setup:** download `creditcard.csv` and place it at `data/creditcard.csv`. See [`data/README.md`](data/README.md). The dataset is **not** committed to this repository.

> The MIT License applies to this project's source code only — not to the third-party Kaggle dataset.

## Project Structure

```text
FraudGuard-ML/
├── data/
│   └── README.md              # dataset download instructions
├── notebooks/
│   └── fraud_detection.ipynb  # exploratory data analysis
├── models/
│   ├── fraud_model.pkl        # trained model + threshold
│   ├── metrics.json           # measured evaluation metrics
│   └── figures/               # PR/ROC/confusion plots
├── train.py
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `pip` fails with `Failed to resolve ... files.pythonhosted.org`, DNS cannot reach the PyPI CDN. Use:

```powershell
python install_deps.py
```

Or fix DNS permanently (**Admin PowerShell**), then retry `pip`:

```powershell
Add-Content $env:SystemRoot\System32\drivers\etc\hosts "`n151.101.0.223 files.pythonhosted.org"
```

Place `creditcard.csv` in the `data/` folder before training.

## Training

```powershell
python train.py
```

This script:

1. Validates and loads the dataset
2. Splits data with stratification
3. Trains Logistic Regression and XGBoost
4. Selects the best model by validation Average Precision
5. Tunes the classification threshold on validation F1
6. Evaluates once on the test set
7. Saves `models/fraud_model.pkl` and metrics/figures under `models/`

## Running the Application

```powershell
streamlit run app.py
```

Upload a CSV with columns `Time`, `V1`–`V28`, and `Amount`. A `Class` column is optional and ignored during prediction.

## Model Performance

Selection rule: highest **validation Average Precision**. Threshold: maximize **validation F1**. Test metrics are reported once and are not used for tuning.

<!-- METRICS_START -->
Validation metrics at threshold `0.5` (used for model comparison / selection by Average Precision):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9786 | 0.0671 | 0.8784 | 0.1246 | 0.9684 | 0.6301 |
| XGBoost | 0.9979 | 0.4485 | 0.8243 | 0.5810 | 0.9820 | **0.8039** |

**Selected model:** XGBoost (highest validation Average Precision)  
**Selected threshold:** 0.9925 (maximizes validation F1 = 0.7761)

Final **test** evaluation (untouched holdout, using the tuned threshold):

| Metric | Value |
|---|---:|
| Accuracy | 0.9994 |
| Precision | 0.9464 |
| Recall | 0.7162 |
| F1-Score | 0.8154 |
| ROC-AUC | 0.9624 |
| Average Precision | 0.8087 |

Test confusion matrix: TN=42,645 · FP=3 · FN=21 · TP=53
<!-- METRICS_END -->

## Screenshots

Add Streamlit screenshots here after running the app:

1. Dashboard / metrics overview
2. Prediction results table
3. Fraud prediction distribution chart

Suggested location: `docs/screenshots/` (optional).

## Limitations

- Trained on a public research dataset, not private bank data
- Features are anonymized (PCA); domain interpretability is limited
- Predictions are estimated fraud risk scores, not definitive labels
- This is a portfolio demonstration, not a production banking system

## License

This project's source code is released under the [MIT License](LICENSE).

The Credit Card Fraud Detection dataset remains subject to its own Kaggle / ULB terms and is **not** covered by this MIT License.
