# Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle (ULB Machine Learning Group).

## Download

1. Visit: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place the file in this directory:

```text
data/creditcard.csv
```

## Notes

- The dataset is **not** included in this repository (it is large and subject to Kaggle's terms).
- Training (`python train.py`) will fail with a clear error message if the file is missing.
- Do not rename the file; the scripts expect `creditcard.csv`.
