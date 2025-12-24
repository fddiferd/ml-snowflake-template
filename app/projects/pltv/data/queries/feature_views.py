

RETENTION_METRICS_QUERY = """
with retention_metrics as (
    -- clean the column names to remove gross add entity
    select
        gross_add__created__month as start_date_month,
        gross_add__brand as brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        gross_add__campaign as campaign,
        gross_add__plan__is_promo as plan__is_promo,
        survived_promo_activations_excl_retries,
        eligible_promo_activations,
        survived_first_rebills_excl_retries,
        eligible_first_rebills,
    from bi_layer_db.prod.exp_pltv_retention_metrics
) 

select
    start_date_month,
    -- group bys
    {group_bys}
    plan__is_promo,
    -- metrics
    sum(eligible_promo_activations) as eligible_promo_activations,
    sum(eligible_first_rebills) as eligible_first_rebills,
    div0(
        sum(survived_promo_activations_excl_retries), 
        sum(eligible_promo_activations)
    )  as promo_activation_rate,
    div0(
        sum(survived_first_rebills_excl_retries), 
        sum(eligible_first_rebills)
    ) as first_rebill_rate,   
from retention_metrics
group by all
"""

BILLING_METRICS_QUERY = """
with billing_metrics as (
    select
        gross_add__created__month as start_date_month,
        gross_add__brand as brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        gross_add__campaign as campaign,
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
    start_date_month,
    -- group bys
    {group_bys}
    plan__is_promo,
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