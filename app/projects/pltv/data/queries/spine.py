QUERY = """
with base as (
    select
        gross_add__created__day,
        brand,
        sku_type,
        channel,
        traffic_source,
        coalesce(campaign, 'NONE') as campaign,
        gross_add__type,
        plan__offer_type,
        plan__is_promo,
        plan__promo_price,
        plan__promo_days,
        plan__recurring_price,
        plan__recurring_days,
        gross_adds,
        gross_adds_canceled_day_one,
        gross_adds_canceled_day_three,
        gross_adds_canceled_day_seven
    from bi_layer_db.prod.exp_pltv_gross_add_metrics
    where gross_add__created__day >= '2021-01-01'
)

select
    date_trunc(month, gross_add__created__day) {timestamp_col},
    {group_bys}
    {partitions}
    -- dims
    sum(case when datediff(day, gross_add__created__day, current_date()) > 1 then gross_adds else 0 end) as gross_adds_created_over_1_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 3 then gross_adds else 0 end) as gross_adds_created_over_3_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 7 then gross_adds else 0 end) as gross_adds_created_over_7_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 30 then gross_adds else 0 end) as gross_adds_created_over_30_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 60 then gross_adds else 0 end) as gross_adds_created_over_60_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 90 then gross_adds else 0 end) as gross_adds_created_over_90_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 180 then gross_adds else 0 end) as gross_adds_created_over_180_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 365 then gross_adds else 0 end) as gross_adds_created_over_365_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 730 then gross_adds else 0 end) as gross_adds_created_over_730_days_ago,
    -- metrics
    sum(gross_adds) as gross_adds,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 1 then gross_adds_canceled_day_one else 0 end) as gross_adds_canceled_day_one,
    div0(sum(case when datediff(day, gross_add__created__day, current_date()) > 1 then gross_adds_canceled_day_one else 0 end), sum(case when datediff(day, gross_add__created__day, current_date()) > 1 then gross_adds else 0 end)) as gross_adds_canceled_day_one_rate,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 3 then gross_adds_canceled_day_three else 0 end) as gross_adds_canceled_day_three,
    div0(sum(case when datediff(day, gross_add__created__day, current_date()) > 3 then gross_adds_canceled_day_three else 0 end), sum(case when datediff(day, gross_add__created__day, current_date()) > 3 then gross_adds else 0 end)) as gross_adds_canceled_day_three_rate,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 7 then gross_adds_canceled_day_seven else 0 end) as gross_adds_canceled_day_seven,
    div0(sum(case when datediff(day, gross_add__created__day, current_date()) > 7 then gross_adds_canceled_day_seven else 0 end), sum(case when datediff(day, gross_add__created__day, current_date()) > 7 then gross_adds else 0 end)) as gross_adds_canceled_day_seven_rate,
    case when plan__is_promo then div0(sum(plan__promo_days * gross_adds), sum(gross_adds)) end as avg_promo_days,
    case when plan__is_promo then div0(sum(plan__promo_price * gross_adds), sum(gross_adds)) end as avg_promo_price,
    div0(sum(plan__recurring_days * gross_adds), sum(gross_adds)) as avg_recurring_days,
    div0(sum(plan__recurring_price * gross_adds), sum(gross_adds)) as avg_recurring_price,
    div0(avg_promo_days, avg_recurring_days) as promo_to_recurring_days_ratio,
    div0(avg_promo_price, avg_recurring_price) as promo_to_recurring_price_ratio,
from base
group by all
order by gross_adds desc
"""