

RETENTION_METRICS_QUERY = """
with retention_metrics as (
    -- clean the column names to remove gross add entity
    select
        gross_add__created__month as {timestamp_col},
        gross_add__brand as brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        coalesce(gross_add__campaign, 'NONE') as campaign,
        gross_add__type,
        gross_add__plan__is_promo as plan__is_promo,
        survived_promo_activations_excl_retries,
        eligible_promo_activations,
        survived_first_rebills_excl_retries,
        eligible_first_rebills,
    from bi_layer_db.prod.exp_pltv_retention_metrics
) 

select
    {timestamp_col},
    {group_bys}
    {partitions}
    -- metrics
    coalesce(sum(eligible_promo_activations), 0) as eligible_promo_activations,
    coalesce(sum(survived_promo_activations_excl_retries), 0) as survived_promo_activations_excl_retries,
    coalesce(div0(
        sum(survived_promo_activations_excl_retries), 
        sum(eligible_promo_activations)
    ), 0)  as promo_activation_rate_excl_retries,
    coalesce(sum(eligible_first_rebills), 0) as eligible_first_rebills,
    coalesce(sum(survived_first_rebills_excl_retries), 0) as survived_first_rebills_excl_retries,
    coalesce(div0(
        sum(survived_first_rebills_excl_retries), 
        sum(eligible_first_rebills)
    ), 0) as first_rebill_rate_excl_retries,   
from retention_metrics
group by all
"""

BILLING_METRICS_QUERY = """
with billing_metrics as (
    select
        gross_add__created__month as {timestamp_col},
        gross_add__brand as brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        coalesce(gross_add__campaign, 'NONE') as campaign,
        gross_add__type,
        gross_add__plan__is_promo as plan__is_promo,
        net_billings_30_days_since_gross_add,
        net_billings_60_days_since_gross_add,
        net_billings_90_days_since_gross_add,
        net_billings_180_days_since_gross_add,
        net_billings_365_days_since_gross_add,
        net_billings_730_days_since_gross_add,
    from bi_layer_db.prod.exp_pltv_billing_metrics
)

select
    {timestamp_col},
    {group_bys}
    {partitions}
    -- metrics
    sum(net_billings_30_days_since_gross_add) as net_billings_30_days,
    sum(net_billings_60_days_since_gross_add) as net_billings_60_days,
    sum(net_billings_90_days_since_gross_add) as net_billings_90_days,
    sum(net_billings_180_days_since_gross_add) as net_billings_180_days,
    sum(net_billings_365_days_since_gross_add) as net_billings_365_days,
    sum(net_billings_730_days_since_gross_add) as net_billings_730_days,
from billing_metrics
group by all
"""


CROSS_SELL_METRICS_QUERY = """
with metrics as (
    -- clean the column names to remove gross add entity
    select
        gross_add__created__month as {timestamp_col},
        gross_add__brand as brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        coalesce(gross_add__campaign, 'NONE') as campaign,
        gross_add__type,
        gross_add__plan__is_promo as plan__is_promo,
        plan__recurring_price,
        cross_sell_adds,
        cross_sell_adds_one_day_since_gross_add,
        cross_sell_adds_three_days_since_gross_add,
        cross_sell_adds_seven_days_since_gross_add,
    from bi_layer_db.prod.exp_pltv_cross_sell_metrics
) 

select
    {timestamp_col},
    {group_bys}
    {partitions}
    -- metrics
    sum(cross_sell_adds) as total_cross_sell_adds,
    coalesce(div0(sum(plan__recurring_price * cross_sell_adds) , total_cross_sell_adds), 0) as avg_cross_sell_price,
    coalesce(sum(cross_sell_adds_one_day_since_gross_add), 0) as cross_sell_adds_one_day_since_gross_add,
    coalesce(sum(cross_sell_adds_three_days_since_gross_add), 0) as cross_sell_adds_three_days_since_gross_add,
    coalesce(sum(cross_sell_adds_seven_days_since_gross_add), 0) as cross_sell_adds_seven_days_since_gross_add,
from metrics
group by all
"""