

## Pre EDA

* **Classification:** Picking which horse will win
* **Multi-class classification:** Multiple horses

---

## EDA

### Figure out our label / Y variable

**`pdsBeaten`**: Pounds the horse was beaten by (based on distance and standard scale)
If:

* Horse A carried 100 lbs and won
* Horse B finished behind with a `pdsBeaten = 10` lbs
  → Then Horse B would need to carry 90 lbs (i.e. 10 lbs less) to match Horse A’s performance.

> If I used `pdsBeaten`, I would probably have to turn it back into a classification problem to get probabilities.

---

If `NMFP` is high and `Position = 1`, that means:

* The horse won
* Against more runners → more impressive win

Higher `BetfairSP` = Less likely to win

* Implied Prob = `1 / BetfairSP`

---

### Drop columns and rows you don't think are good

### Check if column data types are all correct

* Maybe an int is a string, or a datetime is an object — make those conversions

### Rename columns if needed

### Check for duplicates

* Used: `train_data_df.loc[train_data_df.duplicated()]` → none found

---

### Data understanding

* Use: Histogram, Box Plot, Correlation
* Could’ve used: KDE, Pivot table, Violin plots
* **Pairplots** and similar — not useful

---

### Quick decision rules (rule of thumb)

* Table data + many categoricals + nonlinear effects ⇒ go **tree-boost**
* Mostly numeric, linear relationships, few features ⇒ **logistic/linear regression** is okay
* Small tidy numeric dataset with smooth target ⇒ try **kernels/GBR**

---

### Dataset Clues (from description)

| Situation                                                                      | Favors LightGBM / CatBoost                                                       | Hurts linear / kernel methods                                                     |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Mixed feature types (e.g. Prize, DistanceYards, Horse, Trainer, Jockey, Going) | Tree splits handle all with zero scaling; CatBoost encodes categoricals natively | Logistic/SVM need one-hot or embeddings → huge sparse matrix, slower, overfitting |
| Non-linear relations & interactions (e.g. Going × Distance, Age²)              | Captured automatically                                                           | Linear models miss unless engineered; SVM is O(n²)                                |
| Missing values                                                                 | Tree models treat NaNs as "another branch" → no imputation needed                | NaNs break SVM/logistic without preprocessing                                     |
| Many rows (big race data)                                                      | LightGBM handles millions efficiently                                            | SVMs scale poorly; GLMs miss complexity                                           |
| Probabilistic output                                                           | Tree-boosters output calibrated probs (via log loss)                             | SVM needs Platt scaling; one-hot logistic may calibrate poorly                    |
| Per-race grouping                                                              | Add `Race_ID` as group or renormalise logits within race                         | No native support — need post-processing                                          |

---

### Feature: Implied Prob

* Can make implied prob = `1 / BetfairSP`, but GBMs can handle raw BetfairSP directly → this step is optional

---

### Imputation

* **Median**: Safer in general, especially with outliers
* **Mean**: Better for symmetric bell curves

  * Could’ve tested mean, had no time

---

## Could’ve done:

* Feature importance
* Voting classifier
* Stacking classifier
* Could also used MLP didnt have that many features tho and the relationships it spits out are very complex and not interpretable could experiment with it though. also since the outputs for the horses are difference for each race you need  to vary the amount of output neurons for classifcation and machine learning or like manually set some of the neurons to zero.
* Applied Claibrationg like Platt Scaling or something right at the end to get valid probabilities 
---

## Benters factors I didn’t consider or could’ve:

* Add in horse **rest days** (not done)
* Focus more on **consumer opinion**

---

### EDA Observation:

* A horse with **past data** is predicted more accurately than a **new horse** entered into the model

---

