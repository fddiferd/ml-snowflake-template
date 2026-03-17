# VBB -- Value Based Bidding

Predict 180-day net billings for new customers so Google Ads can optimize bids toward higher-value conversions.

## How It Works

Google's Value Based Bidding (VBB) adjusts ad spend based on predicted conversion values rather than treating all conversions equally. We train a model to predict `NET_BILLINGS` (total revenue over 180 days post-signup) for each customer, then pass those predicted values back to Google via offline conversion uploads. Google's bidding algorithm uses these values to allocate budget toward audiences and keywords that produce higher-lifetime-value customers.

## Project Structure

```
app/projects/vbb/
├── constants.py              # Column lists, hyperparams, feature config
├── data/
│   ├── query.py              # SQL query joining spine metrics, plans, payments, zip, usage
│   ├── training.py           # Load, filter, and cache the training set
│   ├── prediction.py         # Load prediction data
│   └── scripts/
│       └── describe_training_data.py   # EDA script for profiling features
├── model/
│   ├── pipeline.py           # VBB-specific preprocessing + shared XGBoost pipeline
│   ├── service.py            # Orchestrator: load data, train, evaluate, save outputs
│   └── output/               # Saved plots, metrics, predictions (gitignored)
└── README.md
```

## Running the Model

```bash
cd app
python -m projects.vbb.model.service
```

This loads the training data (from cache if available), trains the model, evaluates on a 10% holdout, and saves all outputs to `model/output/`.

## Key Design Decisions

### 180-Day Lookback Window

We use `NET_BILLINGS` measured 180 days after signup as the target variable. This is long enough to capture meaningful retention and billing patterns (trial conversions, first renewal, early churn) while short enough to maintain data recency. A shorter window (e.g. 90 days) would miss first renewals for many plan types; a longer window (e.g. 365 days) would sacrifice almost a year of recency in the training set.

### Target Floored at Zero

Negative net billings (from refunds exceeding payments) are floored to zero in the SQL query via `GREATEST(net_billings, 0)`. This prevents the model from being pulled toward predicting negative values, which are not actionable for VBB -- Google cannot use negative conversion values. It also reduces noise from outlier refund events.

### 24-Hour Cancellation Censoring

Customers who cancel within 24 hours are excluded from training (`PREDICTION_CENSOR_HOURS = 24`). These represent immediate buyer's remorse or accidental signups and would not have a GCLID conversion uploaded in time for the model's predictions to influence bidding.

### Log-Transform on Target

`NET_BILLINGS` is heavily right-skewed: most customers pay $0-$50, but some pay $300+. Training on raw dollars causes the model to compress predictions toward the mean and under-predict high-value customers. Applying `log1p()` before training and `expm1()` after prediction spreads the target distribution more evenly, giving the model more resolution in the high-value tail -- exactly where VBB needs it most.

### Target Encoding for High-Cardinality Categoricals

Five columns (`BIN_ISSUING_BANK`, `CAMPAIGN`, `CONNECTION_ISP`, `TRAFFIC_SOURCE`, `CARD_NETWORK_SUBTYPE`) have hundreds of unique values. One-hot encoding them creates 250+ sparse features that fragment importance across many near-zero-contribution dummies. Instead, we target-encode them: replace each category with the mean `NET_BILLINGS` of that category in the training set. This collapses each into a single powerful numerical feature. Unseen categories at prediction time fall back to the global mean.

### Time Features

Month and day-of-week are extracted from `GROSS_ADD__CREATED` to capture seasonal acquisition quality patterns (e.g. holiday signups may have different retention characteristics).

### XGBoost Hyperparameters

Tuned for the 660K-row dataset: 300 trees, depth 8, learning rate 0.05. More trees with a lower learning rate gives better generalization than the defaults (100 trees, depth 6, lr 0.1). Subsample and colsample_bytree at 0.8 provide regularization.

### Spearman Correlation as Primary Metric

For VBB, the model's job is to **rank** customers correctly, not predict exact dollar amounts. Spearman rank correlation directly measures this: if the model consistently ranks a $200 customer above a $50 customer, Google's bidding algorithm will allocate spend correctly even if the predicted values are $150 and $40. R-squared and RMSE are also tracked but are secondary.

### Decile Lift Chart

The decile lift chart is the most actionable visualization for VBB. It groups the test set into 10 buckets by predicted value and shows the mean actual value in each bucket. A monotonically increasing chart confirms the model is useful for bid optimization: predicted-high-value customers genuinely are higher value.

## Feature Categories

| Category | Count | Examples |
|----------|-------|---------|
| Categorical (one-hot) | 18 | `BRAND`, `MAIN_ITEM`, `OFFER_TYPE`, `PAYMENT_TYPE` |
| Numerical | 14 | `RECURRING_PRICE`, `PROMO_DAYS`, `DAY_ONE_REPORT_VIEWS` |
| Boolean | 11 | `IS_PROMO`, `HAS_CROSS_SELL`, `IS_PREPAID` |
| Target-encoded | 5 | `BIN_ISSUING_BANK`, `CAMPAIGN`, `TRAFFIC_SOURCE` |
| Time features | 2 | `GROSS_ADD_MONTH`, `GROSS_ADD_DOW` |

## Data Sources

The training query (`data/query.py`) joins:
- **Spine metrics** -- gross add attributes, day-one report views, 180-day net billings
- **Plans** -- product, offer type, promo terms, recurring pricing
- **Payment options** -- payment type, card network, BIN data (affluence, funding source, prepaid, issuing bank)
- **Zip code demographics** -- household income, home value, poverty rate, unemployment
- **Usage behavior** -- device, OS, connection type, ISP
- **Cross-sell metrics** -- add-on product purchases on day one

## Outputs

After a training run, `model/output/` contains:
- `metrics.txt` -- RMSE, MAE, MAPE, R-squared, Spearman correlation
- `actual_vs_predicted.png` -- scatter plot with R-squared annotation
- `residuals.png` -- residual distribution histogram
- `feature_importances.png` -- top 20 features by gain
- `feature_importances.csv` -- full feature importance table
- `decile_lift.png` -- mean actual value per predicted-value decile
- `test_predictions.csv` -- full test set with predictions
