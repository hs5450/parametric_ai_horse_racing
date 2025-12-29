## Files Overview

* `race_probabilities_main_solution.csv` — Final output file as requested
* `main.ipynb` — Simplified version of the notebook; use this for the key results
* `horse-racingfc3bc24ed8.ipynb` — Original notebook with full thought process (warning: very messy)

---

## Key Points

* **Best Model**: LightGBM

* **Hold-out Log Loss**: `0.3244`

* **Hold-out Brier Score**: `0.0908`

* The **Optuna** tuning section runs, though I couldn’t fully test it due to runtime constraints

* There should be **no data leakage** — I’m \~99% confident. Key leakage-prone variables were dropped, and I was especially careful when handling the K-Fold splits.

---

## Feature Selection Strategy

The goal was to retain as many features as possible unless they were clearly unfit. Features were dropped based on:

* **Histograms**: If extremely concentrated in one bin (suggesting low variance/informativeness).
* **Correlation**: Highly correlated features prompted consideration for removal to reduce redundancy.
* **Logical reasoning**: Features that logically shouldn't affect outcomes were excluded.

### Numeric Features

* `Race_ID`: Dropped (temporarily); had no performance signal.
* `DamsireRating`: Dropped due to low variance (concentrated histogram).
* Others were normalized or left unchanged. Transformation method (log/sqrt) was chosen based on bin spread.
* Correlation and box plots were explored but didn't heavily influence drop/transform decisions.

### Normalize:

* `DistanceYard`
* `Prize`
* `MarketOdds_PreviousRun`
* `MarketOdds_2ndPreviousRun`
* `daysSinceLastRun`
* `Age` (optional)

### Keep:

* `Runners`, `Speed_PreviousRun`, `Speed_2ndPreviousRun`, `TrainerRating`, `JockeyRating`, `SireRating`, `meanRunners`

### Imputation + Flagging:

* `MarketOdds_PreviousRun`, `MarketOdds_2ndPreviousRun`, `TrainerRating`, `JockeyRating` (median imputing + flagging)
* `meanRunners`: Optional drop or median impute + missing flag
* Median used over 0 to preserve central tendency; mean wasn’t used due to skewed distributions

---

## Categorical Features (Objects)

* Light analysis focused on unique value counts and NA rates.
* `Course`, `Going`: Few unique values, encoded with One-Hot or Ordinal Encoding (ODE for experimentation).
* `Trainer`, `Jockey`: Target encoded; manageable unique values.
* `Horse`: Too many unique values. Couldn’t be handled effectively. Tried to engineer around it, ultimately dropped.
* `Race_Time`, `Distance`: Transformed.

---

## Feature Engineering

* `ImpliedProb_prev`: `1 / MarketOdds_PreviousRun`
* `SpeedRank`: Rank of horse's speed within a race using `Race_ID`, based on `Speed_PreviousRun` (no leakage)
* Tried more, but many caused leakage or increased dimensionality without benefit

---

## Model Choice

* Objective: Predict probability of winning → classification task
* Considered RNNs (e.g., LSTM) due to potential data order effects, but dropped due to unclear temporal dependencies and complexity
* Target (`y`): `Position` was converted to binary: 1 (win), 0 (otherwise). Also considered `pdsBeaten` as regression alternative.

### Models Tested

* `SVC`, `LogisticRegression`, `HistGradientBoosting`, `LightGBM`, `CatBoost`
* Best result: LightGBM, log-loss ≈ **0.47**

### Challenge

* GridSearch was computationally expensive due to data shape (50k x 90+ features) and 5-fold CV.
* Initial runs were on local Linux machine → too slow
* Switched to Kaggle servers for better performance
* Optuna or Bayesian optimization planned for better hyperparameter tuning

---

## Probability Calibration

* Initially considered softmax but LightGBM outputs probabilities directly
* Final step: Normalize predicted probabilities **per race** (grouped by `Race_ID`):

  ```python
  prob_norm = prob_raw / prob_raw.groupby(Race_ID).transform("sum")
  ```

---

## Evaluation

* Main metric: **Log Loss**
* Also considered: **Brier Score**
* ROC/AUC not included due to time constraints

---

## Key Challenges and Limitations

* Frequent system crashes → lost time
* Data was hard to interpret and lacked clear signals
* Few features could be obviously dropped
* Feature engineering was trial/error heavy
* Couldn't utilize `Horse` column effectively (too many unique values)
* Categorical-numeric relationships were not deeply explored
* Important CV lesson: `train_test_split(..., stratify=y)` does **not** respect `groups`. Learned to use `GroupKFold` instead.

---

## Final Points
This took me quite a while. While there are some clear and immediate improvements that could be made, as mentioned on the call, this task was originally expected to take just a couple of hours, which I’ve definitely exceeded. I also tried to keep the report as brief and concise as possible and if ive missed something please let me know.



