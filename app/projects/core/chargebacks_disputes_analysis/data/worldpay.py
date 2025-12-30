DISPUTE_QUERY = """
select
    REGEXP_REPLACE(wp."Vantiv Payment ID"::string, '^''+', '') as ref_txn_id,
    -- merchant reference format (005928639 OR TXN_5928639)
    tr.id as txn_id,

    'worldpay' as processor,
    'dispute' as type,
    UPPER(
        REPLACE(
            REPLACE(
                REPLACE(wp."Merchant Name" || ' - ' || wp."MID", 'Inc', 'LLC'),
                ', ', ' '
            ),
            '  ', ' '
        )   
    ) as merchant_account,
    tr.brand_slug,
    tr.channel_slug,
    ts.short_name as traffic_source_short_name,
    tr.bin,
    p.offer_type,
    count(distinct wp."ARN") as count
from pc_fivetran_db.accounts_bi.financial_detail_chargebackfinancialbysettlementdate_mv wp

left join bi_layer_db.prod.transactions tr on
    tr.payment_processor_id = ref_txn_id

left join bi_layer_db.prod.dim_traffic_sources ts on
    ts.id = tr.traffic_source_id

left join bi_layer_db.prod.dim_plans p on
    p.id = tr.plan_id
    
where convert_timezone('America/Los_Angeles', wp."Reporting Date") >= '{date}'
and wp."Cycle" = 'FIRST_CHARGEBACK'
and merchant_account not ilike '%classmates%'
group by all
"""

queries = [
    DISPUTE_QUERY,
]