## Pre EDA
Classification -> Picking which horse will win
Multi Class classfication-> Multiple Horses
## EDA

### Figure out our label our Y variables 
pdsBeaten :Pounds the horse was beaten by (based on distance and standard scale  ---- If:

Horse A carried 100 lbs and won, and

Horse B finished behind with a pdsBeaten = 10 lbs,

Then we say:

Horse B would need to carry 90 lbs (i.e. 10 lbs less) to match Horse A’s performance.

IF i used pdsBeaten i would probably have to trun it back to classifcation problem to get the probaiblites 


If NMFP is high and Position = 1, that means:

The horse won

Against more runners, so it's a more impressive win

Higher BetfairSP = Less likely to win -> Implied Prob = 1 / BetFairSP

### Drop Columns and rows you dont think are good 
### Check if column data types are all good i.e maybe an int is a string or a date_time is an object make that conversion
### Also rename columns in needed Also check duplicated which i have train_data_df.loc[train_data_df.duplicated()] there is non

### Data Understanding and stuff Histogram, Box Plot, correlation ,couldve used KDE, Pivot table, violin plots 

### Pairplots and stuff not usuefl


Quick decision rules (rule-of-thumb)
Table data + many categoricals + nonlinear effects ⇒ go tree-boost.

Mostly numeric, linear relationships, few features ⇒ logistic/linear regression okay.

Small tidy numeric dataset with smooth target ⇒ try kernels/GBR.



Dataset clue (from description)	Favors LightGBM / CatBoost	Hurts linear / kernel methods
Mixed feature types: numbers (Prize, distanceYards), high-cardinality categoricals (Horse, Trainer, Jockey), ordinals (Going)	Tree splits handle all of these with zero scaling; CatBoost even encodes categoricals natively.	Logistic/SVM need one-hot or embeddings ⇒ huge sparse matrix, slower and easy to overfit.
Non-linear relations & interactions are likely: e.g., Going × Distance, Age²	Gradient boosting captures arbitrary interactions automatically.	Linear models miss them unless you manually create terms; SVM captures them but kernel trick is 
𝑂
(
𝑛
2
)
O(n 
2
 ) on big data.
Missing values present	LightGBM/CatBoost treat NaNs as “another branch” → no imputation needed (or done internally).	NaNs break SVM/logistic without preprocessing.
Many rows (“horse-race” data are usually big)	LightGBM is engineered for millions of rows with histogram-based splits.	SVMs scale poorly; GLMs stay fast but miss complexity.
Probabilistic output required	Tree-boosters output calibrated class probabilities (via logistic loss).	Raw SVM scores need Platt scaling; calibration of one-hot-bloated logistic often degrades.
Need per-race grouping	You can add Race_ID as “group” for ranking loss or simply renormalise LightGBM logits within each race.	Traditional classifiers have no grouping notion; you must post-process anyway.




Make Feature: Implied Prob or the betFairSP but its fine for gbm it tanks it it just wastes an extra step


Imputation : Median -> Better in general (more safer espeically whne you haev extreme outliers) , Mean -> better if bell curve is symmetric couldve tested mean had no time.

## COuldve done feature importance but didnt
## Couvle used a voting classfier or a stacking classifier

## Benters Factors i dint consider or couldve 
add in horse rest days whihc i didnt do.
Focus on consumer Opinion the most


## Implimenting EDA
### Horse with past data has a more accurate prediction then a new horse entered into the model



