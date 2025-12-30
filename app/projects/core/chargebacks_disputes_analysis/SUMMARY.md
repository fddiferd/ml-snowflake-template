# Chargebacks & Disputes Risk Analysis

## What We Did

We built an analysis to **identify which areas of the business have higher fraud and chargeback risk**.

## Data Sources

We pulled data from our two payment processors (**Adyen** and **Braintree**) including:
- **Fraud alerts** - early warnings when a card is flagged as potentially fraudulent
- **Chargebacks/Disputes** - when customers contest a charge with their bank
- **Settled transactions** - successful payments (our baseline for calculating rates)

## Enriching the Data

The raw payment processor data only tells us *that* fraud or a chargeback happened. To understand *why* or *where*, we joined it with our internal company data to add business context:

| Column | What It Tells Us |
|--------|------------------|
| **Processor** | Which payment processor handled the transaction (Adyen or Braintree) |
| **Merchant Account** | The specific merchant account used |
| **Card Network** | Visa, Mastercard, Amex, etc. |
| **Brand** | Which brand the customer purchased from |
| **SKU Type** | The type of product/subscription |
| **Channel** | How the customer reached us (web, mobile, etc.) |
| **Traffic Source** | Where the customer came from (Google, Facebook, etc.) |
| **Offer Type** | The pricing/trial offer the customer signed up with |
| **BIN** | The first 6 digits of the credit card (identifies issuing bank) |

## The Analysis

For each dimension above, we calculate the **fraud + dispute rate** (fraud + chargebacks ÷ total settled transactions) and compare it to the overall company average. This tells us:

- Which areas have **higher risk** than average (red flags)
- Which areas have **lower risk** than average (healthy areas)

## Timeframe

Last **6 months** of data.

## Outcome

A prioritized list of high-risk areas that warrant attention, helping teams focus fraud prevention and operational improvements where they'll have the most impact.
