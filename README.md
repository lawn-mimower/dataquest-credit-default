# Credit card default prediction: DataQuest 2025, Round 2

Round 2 of DataQuest 2025 set the "Credit Card Behavior Score" problem. "Bank A" wants the probability that an existing, currently-not-past-due credit card account will default (`bad_flag = 1`). The data is 77,444 anonymised accounts with 1,214 features, and only 1.4% of them default. So the real problem is finding a weak signal in a wide, very imbalanced table. This repository holds our team's notebook: missing-value filtering, correlation-based feature pruning and a class-weighted XGBoost.

Tested honestly on held-out rows, the model separates defaulters only modestly (ROC AUC ≈ 0.69). At the notebook's 0.35 threshold it catches about 5% of them.

**Team "Neurals":** Mihir Mohite, Shreyash Dhoot, Nishad Dere.

## Approach

`Neuralscode.ipynb` runs the whole pipeline.

1. **A look at credit limit.** It creates an `onus_attribute_1 < 500,000` indicator and compares default rates: 1.52% below the cut-off vs 1.17% for the rest. It also plots the distribution by default status.
2. **Missing values.** It drops the 19 columns with more than 50% missing values, makes a stratified 80/20 split and median-imputes.
3. **Correlation pruning**, fitted on the 80% part:
   - For every feature pair with |r| > 0.95, it drops the member less correlated with `bad_flag` (176 features).
   - It then drops features with |r| < 0.017 against `bad_flag` (763 features).
   - This leaves 257 features.
   - The correlation matrix is computed with RAPIDS cuDF when it is installed, and with pandas otherwise.
4. **Train/test split.** It pools the rows again and samples 45% of defaulters and 55% of non-defaulters for training. The rows not sampled are the test set. Features are standardised.
5. **Model.** An `XGBClassifier` with default hyperparameters and `scale_pos_weight` = non-defaulters / defaulters ≈ 69. The classification metrics use a 0.35 threshold on the predicted probability.
6. **Submission.** It prepares `test_data.csv` similarly (impute, drop the pruned features, align columns, scale) and writes `credit_default_predictions.csv` with `account_number,predicted_probability`.

`eda/dataexp.ipynb` loads the training data and computes the full correlation matrix. It also defines a `run_quick_eda()` helper, which it doesn't call. `eda/graphs/` holds saved EDA figures:
- correlation matrices of 250 and 500 randomly sampled features;
- histograms of 10–100 random features;
- a bivariate plot of 10 features against `bad_flag`;
- the "reduced" correlation matrix drawn by `Neuralscode.ipynb`.

The notebooks here don't contain the code for most of the sampled-feature figures.

## Data

The data is **not included**. It was provided to participants by the competition. Put the files in `data/` at the repository root, or point `DATA_DIR` at their folder.

| File | Shape | Contents |
|---|---|---|
| `train_data.csv` | 77,444 × 1,216 | `account_number`, `bad_flag` (1,102 defaults, 1.42%) and 1,214 anonymised features |
| `test_data.csv` | 19,362 × 1,215 | Same columns without `bad_flag` |

The feature groups, as the problem statement describes them:

| Group | Count | Examples |
|---|---|---|
| `onus_attribute_*` | 48 | on-us attributes such as credit limit |
| `transaction_attribute_*` | 664 | number and rupee value of transactions by merchant type |
| `bureau_*` | 452 | bureau tradeline attributes such as product holdings and past delinquencies |
| `bureau_enquiry_*` | 50 | bureau enquiries, e.g. personal-loan enquiries in the last 3 months |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add cudf-cu12 only if you want the GPU correlation step

# Headless run; DATA_DIR must contain train_data.csv and test_data.csv
DATA_DIR=/abs/path/to/data jupyter nbconvert --to notebook --execute \
  --output-dir /tmp/runs Neuralscode.ipynb
```

In our CPU runs:
- `Neuralscode.ipynb` took about 2–3 minutes on the full data. It writes `credit_default_predictions.csv` (19,362 rows) into the working directory.
- `eda/dataexp.ipynb` defaults to `../data`, the same `data/` folder when run from `eda/`, and took about 2.5 minutes.

## Results

These results come from running the notebook as committed on the full `train_data.csv`, on CPU with pandas (no cuDF) and the notebook's fixed random seeds. The test set is the 34,961 rows not sampled for training, 607 of which are defaults (1.74%).

| Metric (threshold 0.35) | Value |
|---|---|
| ROC AUC | 0.695 |
| Recall | 0.048 (29 of 607 defaulters) |
| Precision | 0.080 (29 of 362 flagged) |
| F1 | 0.060 |
| Accuracy | 0.974, *below* the 0.983 you'd get by predicting "no default" for everyone |

A rerun with a diagnostics cell appended (same metrics) showed:
- **Ranking.** The top 10% of test scores contain 28.7% of the defaulters, where random ranking would give 10%. Average precision is 0.047 against a 0.017 base rate.
- **Threshold.** 0.35 is far too high for these scores. Lowering it to 0.05 flags 3,609 accounts and raises recall to 29%, with precision at 4.9%.
- **Overfitting.** The model fits the training data perfectly (training AUC 1.000; all 495 training defaulters score above 0.35) and generalises weakly.

**Why the model is weak.**
- With 1.4% positives and a weight of about 69 on each of the 495 training defaulters, default-depth XGBoost with 100 trees memorises them.
- Pruning by *linear* correlation with a rare binary target discards most columns (939 of the 1,196 candidates) on a very weak criterion.
- No hyperparameters, regularisation or threshold were tuned on validation data.

The honest conclusion: there is some signal (AUC ≈ 0.69, about 2.9× lift in the top decile), but this pipeline doesn't turn it into a usable classifier.

An earlier version of the notebook had two evaluation bugs:
- Roughly half of its test rows were also training rows.
- The submission was scored on unscaled features.

Both are fixed. Figures produced with the overlapping split, an ROC curve and a threshold sweep, have been removed from `eda/graphs/`.

## Known issues and limitations

- **Mild leakage into feature selection.** Imputation and correlation pruning are fitted on the first 80/20 split, but the final test rows are drawn after re-pooling. About 80% of the test rows therefore contributed to choosing the features. `scale_pos_weight` is also computed on all rows.
- **Shifted test mix.** Because of the 45%/55% sampling, the test set has a higher default rate (1.74%) than the data (1.42%), which affects precision and F1.
- **The `onus_attribute_1` indicator is explored but not used.** The modelling pipeline re-reads the CSV, so the indicator never reaches the model.
- **Submission preprocessing differs from training.** `test_data.csv` is median-imputed with its *own* medians, and any training feature missing after preprocessing is filled with 0.
- **Uncalibrated probabilities.** With class weighting, the scores are not calibrated probabilities (mean 0.023 against a 0.017 test default rate). The competition scored how close probabilities were to outcomes, so calibration would matter.
- **No robustness checks.** It uses one random split, with no cross-validation and default hyperparameters.

## Repository layout

```
Neuralscode.ipynb    full pipeline: filtering, correlation pruning, class-weighted XGBoost, submission
eda/dataexp.ipynb    correlation matrix and a quick-EDA helper
eda/graphs/          saved EDA figures
requirements.txt
```

Licence: not yet specified.
