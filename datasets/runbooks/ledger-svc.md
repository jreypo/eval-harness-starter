# ledger-svc online API

ledger-svc is the online ledger API in namespace finance. Checkout and the mobile app call it to read account balances in real time. It keeps a Redis read-through cache of balances in front of Postgres.

## Symptoms of stale balance reads

Customers see an old balance right after a payment. The ledger-svc metric `balance_cache_age_seconds` climbs above 60, usually after a Redis failover or a bulk correction applied directly to Postgres.

## Flush the balance cache

Invalidate the cached balances with `ledgerctl svc cache flush --scope balances`. The service repopulates entries from Postgres on the next read. Do not restart the ledger-svc pods to clear it; the cache lives in Redis and survives restarts.

## Verify the fix

Read a known account through the API with `ledgerctl svc balance get <account>` and compare it with the Postgres value. Watch `balance_cache_age_seconds` drop back under 5.
