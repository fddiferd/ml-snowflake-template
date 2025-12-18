training_spine = """
    select
        -- keys
        date_trunc(month, metric_time__day) as date_month,
        brand,
        sku_type,
        channel,
        traffic_source,
        coalesce(campaign, 'unknown') as campaign,
        plan__offer_type as plan__offer_type,
        coalesce(plan__promo_price, 0) as plan__promo_price,
        coalesce(plan__promo_days, 0) as plan__promo_days,
        plan__recurring_price as plan__recurring_price,
        plan__recurring_days as plan__recurring_days,
        
        sum(gross_adds) as gross_adds

    from BI_LAYER_DB.STAGING.exp_pltv_spine
    where datediff(day, metric_time__day, current_date()) > 730
    group by all
    having sum(gross_adds) > 200
"""

training_metrics = """
    select
        -- keys
        date_trunc(month, gross_add__created__day) as date_month,
        brand,
        gross_add__sku_type as sku_type,
        gross_add__channel as channel,
        gross_add__traffic_source as traffic_source,
        coalesce(gross_add__campaign, 'unknown') as campaign,
        gross_add__plan__offer_type as plan__offer_type,
        coalesce(gross_add__plan__promo_price, 0) as plan__promo_price,
        coalesce(gross_add__plan__promo_days, 0) as plan__promo_days,
        gross_add__plan__recurring_price as plan__recurring_price,
        gross_add__plan__recurring_days as plan__recurring_days,

        -- cross sells
        sum(cross_sell_adds) as cross_sells,
        div0(sum(case when cross_sell_adds > 0 then datediff(day, gross_add__created__day, metric_time__day) * cross_sell_adds else 0 end), cross_sells) as avg_days_to_cross_sell,
        div0(sum(plan__recurring_price * cross_sell_adds), sum(cross_sell_adds)) as avg_cross_sell_price,

        -- retention rates
        case when coalesce(gross_add__plan__promo_price, 0) != 0 and coalesce(gross_add__plan__promo_days, 0) != 0 then div0(sum(survived_promo_activations), sum(eligible_promo_activations)) end as promo_activation_rate,
        div0(sum(survived_first_rebills), sum(eligible_first_rebills)) as first_rebill_rate,
        
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 30 then net_billings else 0 end) as net_billings_30_days,
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 60 then net_billings else 0 end) as net_billings_60_days,
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 90 then net_billings else 0 end) as net_billings_90_days,
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 180 then net_billings else 0 end) as net_billings_180_days,
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 365 then net_billings else 0 end) as net_billings_365_days,
        sum(case when datediff(day, gross_add__created__day, metric_time__day) <= 730 then net_billings else 0 end) as net_billings_730_days,

    from BI_LAYER_DB.STAGING.exp_pltv_metrics
    where datediff(day, gross_add__created__day, current_date()) > 730
    group by all
"""