select
    date_trunc(month, gross_add__created__day) start_date_month,
    -- group bys
    brand,
    sku_type,
    channel,
    -- dims
    sum(case when datediff(day, gross_add__created__day, current_date()) > 30 then gross_adds else 0 end) as gross_adds_created_over_30_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 60 then gross_adds else 0 end) as gross_adds_created_over_60_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 90 then gross_adds else 0 end) as gross_adds_created_over_90_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 180 then gross_adds else 0 end) as gross_adds_created_over_180_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 365 then gross_adds else 0 end) as gross_adds_created_over_365_days_ago,
    sum(case when datediff(day, gross_add__created__day, current_date()) > 730 then gross_adds else 0 end) as gross_adds_created_over_730_days_ago,
    -- metrics
    sum(gross_adds) as gross_adds,
    div0(
        sum(case when plan__offer_type <> 'standard' then plan__promo_days * gross_adds else 0 end),
        sum(gross_adds)
    ) as avg_promo_days,
    div0(
        sum(case when plan__offer_type <> 'standard' then plan__promo_price * gross_adds else 0 end),
        sum(gross_adds)
    ) as avg_promo_price,
    div0(
        sum(plan__recurring_days * gross_adds),
        sum(gross_adds)
    ) as avg_recurring_days,
    div0(
        sum(plan__recurring_price * gross_adds),
        sum(gross_adds)
    ) as avg_recurring_price,
    div0(
        avg_promo_days, 
        avg_recurring_days
    ) as promo_to_recurring_days_ratio,
    div0(
        avg_promo_price, 
        avg_recurring_price
    ) as promo_to_recurring_price_ratio,
from exp_pltv_gross_add_metrics
where start_date_month = '2025-11-01'
group by all
order by gross_adds desc


;


select
    gross_add__created__month as start_date_month,
    -- group bys
    gross_add__brand as brand,
    gross_add__sku_type as sku_type,
    gross_add__channel as channel,
    -- metrics
    sum(eligible_promo_activations) as eligible_promo_activations,
    sum(eligible_first_rebills) as eligible_first_rebills,
    div0(
        sum(survived_promo_activations), 
        sum(eligible_promo_activations)
    )  as promo_activation_rate,
    div0(
        sum(survived_first_rebills), 
        sum(eligible_first_rebills)
    ) as first_rebill_rate,   
from exp_pltv_retention_metrics
where start_date_month = '2025-01-01'
group by all


;


select
    gross_add__created__month as start_date_month,
    -- group bys
    gross_add__brand as brand,
    gross_add__sku_type as sku_type,
    gross_add__channel as channel,
    -- metrics
    sum(net_billings_30_days_since_gross_add) as net_billings_30_days,
    sum(net_billings_60_days_since_gross_add) as net_billings_60_days,
    sum(net_billings_90_days_since_gross_add) as net_billings_90_days,
    sum(net_billings_180_days_since_gross_add) as net_billings_180_days,
    sum(net_billings_365_days_since_gross_add) as net_billings_365_days,
    sum(net_billings_730_days_since_gross_add) as net_billings_730_days,
from exp_pltv_billing_metrics
where start_date_month = '2025-01-01'
group by all;
