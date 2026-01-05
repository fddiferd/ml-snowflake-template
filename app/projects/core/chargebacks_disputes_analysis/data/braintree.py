DISPUTE_QUERY = """
    select
        bt.id as ref_txn_id,
        -- merchant reference format (005928639 OR TXN_5928639)
        tr.id as txn_id,

        'braintree' as processor,
        'dispute' as type,
        bt.merchantaccountid as merchant_account,
        bt.transaction:paymentInstrumentSubtype::string as card_network,
        tr.brand_slug,
        tr.sku_type_slug,
        tr.channel_slug,
        ts.short_name as traffic_source_short_name,
        tr.bin,
        p.offer_type,
        count(distinct bt.id) as count
    from pc_fivetran_db.accounts_bi.braintree_daily_disputes_by_received_date_mv bt

    left join bi_layer_db.prod.transactions tr on
        tr.payment_processor_id = bt.id

    left join bi_layer_db.prod.dim_traffic_sources ts on
        ts.id = tr.traffic_source_id

    left join bi_layer_db.prod.dim_plans p on
        p.id = tr.plan_id
        
    where convert_timezone('America/Los_Angeles', bt.receiveddate) >= '{date}'
    and bt.merchantaccountid <> 'classmates_instant'
    and bt.kind = 'chargeback'
    and bt.transaction:paymentInstrumentSubtype::string <> 'paypal_account'
    group by all
"""

FRAUD_QUERY = """
    select
        bt."transaction_id" as ref_txn_id,
        -- merchant reference format (005928639 OR TXN_5928639)
        tr.id as txn_id,

        'braintree' as processor,
        'fraud' as type,
        bt."merchant_account" as merchant_account,
        bt."payment_instrument_subtype" as card_network,
        tr.brand_slug,
        tr.sku_type_slug,
        tr.channel_slug,
        ts.short_name as traffic_source_short_name,
        tr.bin,
        p.offer_type,
        count(distinct bt."transaction_id") as count
    from pc_fivetran_db.accounts_bi.braintree_tc40_mv bt

    left join bi_layer_db.prod.transactions tr on
        tr.payment_processor_id = bt."transaction_id"

    left join bi_layer_db.prod.dim_traffic_sources ts on
        ts.id = tr.traffic_source_id

    left join bi_layer_db.prod.dim_plans p on
        p.id = tr.plan_id
        
    where convert_timezone('America/Los_Angeles', bt."fraud_post_date") >= '{date}'
    and bt."merchant_account" <> 'classmates_instant'
    group by all
"""

SETTLEMENT_QUERY = """
    select
        bt.id as ref_txn_id,
        -- merchant reference format (005928639 OR TXN_5928639)
        tr.id as txn_id,

        'braintree' as processor,
        'settlement' as type,
        bt.merchantaccountid as merchant_account,
        po.card_network,
        tr.brand_slug,
        tr.sku_type_slug,
        tr.channel_slug,
        ts.short_name as traffic_source_short_name,
        tr.bin,
        p.offer_type,
        count(distinct bt.id) as count
    from pc_fivetran_db.accounts_bi.braintree_daily_transactions_detail_mv bt

    left join bi_layer_db.prod.transactions tr on
        tr.payment_processor_id = bt.id

    left join bi_layer_db.prod.payment_options po on
        po.id = tr.payment_option_id

    left join bi_layer_db.prod.dim_traffic_sources ts on
        ts.id = tr.traffic_source_id

    left join bi_layer_db.prod.dim_plans p on
        p.id = tr.plan_id
        
    where convert_timezone('America/Los_Angeles', bt.createdat) >= '{date}'
    and bt.merchantaccountid <> 'classmates_instant'
    and bt.type = 'sale'
    and value:paymentInstrumentType <> 'paypal_account'
    group by all
"""

queries = [
    DISPUTE_QUERY,
    FRAUD_QUERY,
    SETTLEMENT_QUERY,
]