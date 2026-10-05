# Postgres primary failover

Promote the streaming replica when the Postgres primary for the orders database is lost and Patroni has not failed over on its own.

## When to use

Alert `PostgresPrimaryDown` has fired for more than 2 minutes and `patronictl list` shows no leader. If Patroni already elected a new leader, do not intervene; just verify application connectivity.

## Check replication lag before promoting

Run `patronictl list` and note the lag column for each replica. Pick the replica with the lowest lag. If every replica lags by more than 30 seconds of WAL, page the database owner before promoting, because the promotion will lose those transactions.

## Fence the old primary

Before promoting anything, make sure the old primary cannot accept writes. Stop its pod or shut the instance down, and remove it from the `orders-db-rw` service endpoints. Skipping fencing risks split brain, where two primaries accept conflicting writes.

## Promote the replica

Run `patronictl failover --candidate <replica> --force`. Patroni promotes the replica, updates the leader key and repoints the `orders-db-rw` service. Applications using the service DNS name reconnect on their next retry.

## Verify and rebuild

Confirm writes succeed with a test insert into the `healthcheck` table. Then rebuild the old primary as a replica with `patronictl reinit orders-db <old-member>` so the cluster has redundancy again.
