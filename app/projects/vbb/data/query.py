QUERY = """--sql
select
    -- passthrough columns
    m.gross_add,
    m.gross_add__created_time__second as gross_add__created,
    m.gross_add__hours_to_cancel,
    c.email,
    c.phone,
    m.gross_add__cake_last_click_conversion__sub_id_4 as gclid,
    -- gross add
    m.gross_add__brand as brand,
    m.gross_add__type,
    m.gross_add__level,
    -- source
    m.gross_add__channel as channel,
    m.gross_add__traffic_source as traffic_source,
    m.gross_add__campaign as campaign,
    -- customer 
    -- cdi.birth_year,
    -- usage behavior purchase (gross add)
    --ubp.device_is_mobile,
    ubp.device_os_name,
    -- ubp.device_platform_model,
    -- ubp.device_platform_vendor,
    ubp.device_is_os_apple,
    ubp.device_is_os_android,
    -- ubp.connection_time_zone,
    ubp.connection_user_type,
    ubp.connection_isp,
    ubp.connection_type,
    ubp.connection_is_anonymous,
    -- payment option
    po.payment_type,
    po.payment_type != 'paypal' as has_card_data,
    po.card_network,
    po.card_network_subtype,
    po.is_prepaid,
    po.bin_affluence,
    po.bin_issuing_bank,
    po.bin_issuer_country,
    po.bin_funding_source,
    po.bin_corporate,
    po.bin_prepaid_type,
    po.bin_reloadable,
    po.avs_result_type,
    -- zip
    z.median_household_income,
    z.per_capita_income,
    z.median_home_value,
    z.poverty_rate,
    z.unemployment_rate,
    -- plan 
    p.main_item,
    p.is_promo,
    p.offer_type,
    coalesce(p.promo_days, 0) as promo_days,
    coalesce(p.promo_price, 0) as promo_price,
    p.recurring_days,
    p.recurring_price,
    p.annual_recurring_price,
    -- report views
    coalesce(m.day_one_gross_add_report_views, 0) as day_one_report_views,
    day_one_report_views > 0 as has_day_one_report_view,
    coalesce(m.day_one_gross_add_self_search_report_views, 0) as day_one_self_search_report_views,
    day_one_self_search_report_views > 0 as has_day_one_self_search_report_view,
    coalesce(m.day_one_gross_add_family_search_report_views, 0) as day_one_family_report_views,
    day_one_family_report_views > 0 as has_day_one_family_report_view,
    -- cross sells
    coalesce(cm.cross_sell_adds, 0) as cross_sell_adds,
    coalesce(cm.cross_sell_adds, 0) > 0 as has_cross_sell,
    -- target
    greatest(m.net_billings_180_days_since_gross_add, 0) as net_billings,
    
from bi_layer_db.prod.exp_vbb_spine_metrics m
left join bi_layer_db.prod.exp_vbb_cross_sell_metrics cm on cm.gross_add = m.gross_add
join bi_layer_db.prod.customers c on c.id = m.gross_add__customer
join bi_layer_db.prod.dim_plans p on p.id = m.gross_add__plan
join bi_layer_db.prod.payment_options po on po.id = m.gross_add__payment_option
left join bi_layer_db.prod.zip_code_demo z on z.zip_code = po.zip_code
left join bi_layer_db.prod.latest_customer_birth_year_inputs cdi on cdi.customer_id = m.gross_add__customer
left join bi_layer_db.prod.usage_behavior_purchases ubp on ubp.gross_add_id = m.gross_add
where true
    and m.gross_add__created_time__second > '{from_time}'
    and m.gross_add__created_time__second <= '{to_time}'
--endsql"""