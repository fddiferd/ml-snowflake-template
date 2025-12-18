from pltv.db import SnowflakeContext
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import seaborn as sns
import json
import re
import logging
from tqdm.contrib.concurrent import process_map
from tqdm.notebook import tqdm


plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)

plt.rcParams["figure.figsize"] = (20, 10)

levels = [
    "brand",
    "acquisition_sku_bucket",
    "acquisition_channel",
    "payment_type",
    'mobile',
]

def get_data(levels:list[str]):
    query = """
      SELECT
      DATE(DATE_TRUNC(day,acquisition_date)) as acquisition_date,
      {levels},
      count(distinct customer_id) as cohort_size,
      COUNT(DISTINCT CASE WHEN canceled_day_1 = 1 THEN customer_id ELSE null END )/cohort_size as canceled_day_1,
      SUM(case when ((month_age) * (recurring_period_days/30)) < 1 THEN 
         COALESCE(INITIAL_SUB_AMOUNT,0) 
         +COALESCE(CROSS_AMOUNT,0)
         +COALESCE(UPSELL_AMOUNT,0)
         +COALESCE(RESUB_AMOUNT,0)
         +COALESCE(DOWNSELL_AMOUNT,0)
         -COALESCE(REFUND_AMOUNT,0)
         ELSE 0 END
      )/cohort_size as ltv_1mo,
      SUM(case when ((month_age) * (recurring_period_days/30)) < 3 THEN 
         COALESCE(INITIAL_SUB_AMOUNT,0) 
         +COALESCE(CROSS_AMOUNT,0)
         +COALESCE(UPSELL_AMOUNT,0)
         +COALESCE(RESUB_AMOUNT,0)
         +COALESCE(DOWNSELL_AMOUNT,0)
         -COALESCE(REFUND_AMOUNT,0)
         ELSE 0 END
      )/cohort_size as ltv_3mo,
      SUM(case when ((month_age) * (recurring_period_days/30)) < 6 THEN 
         COALESCE(INITIAL_SUB_AMOUNT,0) 
         +COALESCE(CROSS_AMOUNT,0)
         +COALESCE(UPSELL_AMOUNT,0)
         +COALESCE(RESUB_AMOUNT,0)
         +COALESCE(DOWNSELL_AMOUNT,0)
         -COALESCE(REFUND_AMOUNT,0)
         ELSE 0 END
      )/cohort_size as ltv_6mo,
      SUM(case when ((month_age) * (recurring_period_days/30)) < 12 THEN 
         COALESCE(INITIAL_SUB_AMOUNT,0) 
         +COALESCE(CROSS_AMOUNT,0)
         +COALESCE(UPSELL_AMOUNT,0)
         +COALESCE(RESUB_AMOUNT,0)
         +COALESCE(DOWNSELL_AMOUNT,0)
         -COALESCE(REFUND_AMOUNT,0)
         ELSE 0 END
      )/cohort_size as ltv_12mo,
      SUM(case when ((month_age) * (recurring_period_days/30)) < 24 THEN 
         COALESCE(INITIAL_SUB_AMOUNT,0) 
         +COALESCE(CROSS_AMOUNT,0)
         +COALESCE(UPSELL_AMOUNT,0)
         +COALESCE(RESUB_AMOUNT,0)
         +COALESCE(DOWNSELL_AMOUNT,0)
         -COALESCE(REFUND_AMOUNT,0)
         ELSE 0 END
      )/cohort_size as ltv_24mo,
   FROM pltv.vbb.customer_billings cb
   group by all order by 1, cohort_size desc
   
   """.format(levels=",\n".join(levels))
    with SnowflakeContext() as ctx:
        raw_df = ctx.fetch(query)
        raw_df.acquisition_date = pd.to_datetime(raw_df.acquisition_date)
        return raw_df


logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def forecast(row):
    columns = ["canceled_day_1", "ltv_1mo", "ltv_3mo", "ltv_6mo", "ltv_12mo"]
    results = {}
    full = row.copy()
    full["acquisition_date"] = pd.to_datetime(full["acquisition_date"])
    full.set_index("acquisition_date", inplace=True)

    for idx, target in enumerate(columns):
        blackout_days = (int(re.sub(r"\D", "", target)) * 30) + 45
        train = row.rename(columns={"acquisition_date": "ds", target: "y"})
        train.y = train.y.astype(float)
        train.ds = pd.to_datetime(train.ds)

        # date_range = pd.date_range(
        #     start=train.ds.min(), end=train.ds.max(), freq="d"
        # )

        # train = train.reindex(date_range)

        train = train[(train.ds <= train.ds.max() - pd.DateOffset(days=blackout_days))]

        train["cap"] = train.y.astype(float).max() * 1.2
        train["floor"] = 0

        for col in columns[:idx]:
            if idx == 0:
                continue
            train[col] = row[col].astype(float).fillna(0)
        m = Prophet(changepoint_prior_scale=0.82, growth="logistic")

        for col in columns[:idx]:
            if idx == 0:
                continue
            m.add_regressor(col)
        try:
            m.fit(train)
        except Exception as e:
            return train, None

        future = m.make_future_dataframe(periods=480, freq="d", include_history=True)
        for col in columns[:idx]:
            if idx == 0:
                continue
            future = future.merge(
                right=full[f"{col}_predicted"].rename(col),
                left_on="ds",
                right_index=True,
                how="inner",
            )

        future["cap"] = train.y.astype(float).max() * 1.2
        future["floor"] = 0
        forecast = m.predict(future)

        combined = forecast.merge(
            train[["ds", "y"]].rename(columns={"y": "actuals"}),
            right_on="ds",
            left_on="ds",
            how="outer",
        )
        full = (
            full.merge(
                forecast[["ds", "yhat"]].rename(
                    columns={"yhat": f"{target}_predicted"}
                ),
                how="left",
                left_index=True,
                right_on="ds",
            )
            .rename(columns={"ds": "acquisition_date"})
            .set_index("acquisition_date")
        )

        results[target] = {
            "forecast": combined,
            "model": m,
        }
    return full, results


# test_grp = dict(
#     brand ='Intelius',
#     acquisition_sku_bucket = 'phone_report-trial-30',
#     payment_type = 'paypal',
# )
# df = df[
#     (df.brand == test_grp['brand'])
#     & (df.acquisition_sku_bucket == test_grp['acquisition_sku_bucket'])
#     & (df.payment_type == test_grp['payment_type'])
#     & (df.acquisition_channel == 'Paid Search')
# ]
# df.acquisition_date = pd.to_datetime(df.acquisition_date)
# full, model = forecast(df)

all_results = pd.DataFrame()


for idx, _ in enumerate(levels[1:]):
    level = levels[: idx + 2]
    raw_df = get_data(level)
    df = raw_df.copy()
    df.acquisition_date = pd.to_datetime(df.acquisition_date)

    for grp, row in df.groupby(level):
        filters = dict(zip(level, grp))
        print(f"Forecasting {filters}")
        if len(row) == 0 or row.cohort_size.sum() <= 50:
            continue
        try:
            full, model = forecast(row)
            if model is None:
                break
            full = full[
                [
                    "cohort_size",
                    "canceled_day_1",
                    "ltv_1mo",
                    "ltv_3mo",
                    "ltv_6mo",
                    "ltv_12mo",
                    "ltv_24mo",
                    "ltv_1mo_predicted",
                    "ltv_3mo_predicted",
                    "ltv_6mo_predicted",
                    "ltv_12mo_predicted",
                ]
            ]
            full["filters"] = json.dumps(filters)
            full["level"] = level[-1]
            all_results = pd.concat([full, all_results])
        except Exception as e:
            print(f"Error on {filters}")
            print(e)
            continue


with SnowflakeContext() as ctx:
    to_write = all_results.reset_index()
    to_write.acquisition_date = to_write.acquisition_date.dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    ctx.write_pandas(
        df=to_write,
        database="PLTV",
        schema="VBB",
        table="PLTV_RESULTS",
        auto_create_table=True,
        overwrite=True,
    )