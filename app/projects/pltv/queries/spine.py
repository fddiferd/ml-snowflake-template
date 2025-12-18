QUERY = """
select
    date_trunc(month, gross_add__created__day) start_date_month,
    -- group bys
    {group_bys}
    plan__is_promo,
    -- dims
    sum(case when datediff(day, gross_add__created__day, current_date()) > 30 then gross_adds else 0 end) as gross_adds_created_over_30_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 60 then gross_adds else 0 end) as gross_adds_created_over_60_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 90 then gross_adds else 0 end) as gross_adds_created_over_90_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 180 then gross_adds else 0 end) as gross_adds_created_over_180_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 365 then gross_adds else 0 end) as gross_adds_created_over_365_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 730 then gross_adds else 0 end) as gross_adds_created_over_730_days_ago,
    -- metrics
    sum(gross_adds) as gross_adds,
    case when plan__is_promo then div0(sum(plan__promo_days * gross_adds), sum(gross_adds)) end as avg_promo_days,
    case when plan__is_promo then div0(sum(plan__promo_price * gross_adds), sum(gross_adds)) end as avg_promo_price,
    div0(sum(plan__recurring_days * gross_adds), sum(gross_adds)) as avg_recurring_days,
    div0(sum(plan__recurring_price * gross_adds), sum(gross_adds)) as avg_recurring_price,
    div0(avg_promo_days, avg_recurring_days) as promo_to_recurring_days_ratio,
    div0(avg_promo_price, avg_recurring_price) as promo_to_recurring_price_ratio,
from bi_layer_db.dbt_donato.exp_pltv_gross_add_metrics
where start_date_month >= '2021-11-01'
group by all
order by gross_adds desc
"""