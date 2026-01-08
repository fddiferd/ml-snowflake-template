FRAUD_DISPUTE_QUERY = """
    select
        "Capture Modification Merchant Reference" as ref_txn_id,
        -- merchant reference format (005928639 OR TXN_5928639)
        IFF(
            a."Capture Modification Merchant Reference" LIKE 'TXN_%',
            SUBSTR(a."Capture Modification Merchant Reference", 5),
            a."Capture Modification Merchant Reference"
        )::int as txn_id,

        'adyen' as processor,
        case
            when a."Record Type" = 'Chargeback' then 'dispute' 
            when a."Record Type" = 'NotificationOfFraud' then 'fraud'
        end as type,
        a."Merchant Account" as merchant_account,
        a."Payment Method" as card_network,
        tr.brand_slug,
        tr.sku_type_slug,
        tr.channel_slug,
        ts.short_name as traffic_source_short_name,
        tr.bin,
        p.offer_type,
        count(distinct txn_id) as count

    from pc_fivetran_db.accounts_bi.adyen_dispute_report_mv a

    left join bi_layer_db.prod.transactions tr on
        tr.id = txn_id

    left join bi_layer_db.prod.dim_traffic_sources ts on
        ts.id = tr.traffic_source_id

    left join bi_layer_db.prod.dim_plans p on
        p.id = tr.plan_id
        
    where convert_timezone('America/Los_Angeles', "Record Date") >= '{date}'
    and a."Merchant Account" <> 'ClassmatesECOM'
    and not a."RDR"
    and a."Record Type" in ('Chargeback', 'NotificationOfFraud')
    group by all
"""

SETTLEMENT_QUERY = """
    select
        "Merchant Reference" as ref_txn_id,
        -- merchant reference format (005928639 OR TXN_5928639)
        IFF(
            a."Merchant Reference" LIKE 'TXN_%',
            SUBSTR(a."Merchant Reference", 5),
            a."Merchant Reference"
        )::int as txn_id,

        'adyen' as processor,
        'settlement' as type,
        a."Merchant Account" as merchant_account,
        a."Payment Method" as card_network,
        tr.brand_slug,
        tr.sku_type_slug,
        tr.channel_slug,
        ts.short_name as traffic_source_short_name,
        tr.bin,
        p.offer_type,
        count(distinct txn_id) as count

    from pc_fivetran_db.accounts_bi.adyen_payments_accounting_report_mv a

    left join bi_layer_db.prod.transactions tr on
        tr.id = txn_id

    left join bi_layer_db.prod.dim_traffic_sources ts on
        ts.id = tr.traffic_source_id

    left join bi_layer_db.prod.dim_plans p on
        p.id = tr.plan_id
        
    where convert_timezone('America/Los_Angeles', "Creation Date") >= '{date}'
    and a."Merchant Account" <> 'ClassmatesECOM'
    and a."Record Type" in ('Settled')
    group by all
"""

queries = [
    FRAUD_DISPUTE_QUERY,
    SETTLEMENT_QUERY,
]