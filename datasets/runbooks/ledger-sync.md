# ledger-sync batch job

ledger-sync is the nightly CronJob in namespace finance that copies settled ledger entries into the analytics warehouse. It does not serve user traffic. Stale balances in the warehouse dashboards come from this job, not from the online ledger service.

## Symptoms of a failed sync

The finance dashboards show stale balances from the previous day, and alert `LedgerSyncLastSuccessTooOld` fires when the last successful sync is older than 26 hours.

## Reset the sync cursor

A failed sync leaves the cursor pointing at a partially copied batch. Reset the sync cursor to the last committed batch with `ledgerctl sync cursor reset --to last-committed`, then re-run the job manually with `kubectl -n finance create job ledger-sync-manual --from=cronjob/ledger-sync`.

## Verify the sync

Watch the manual job logs until it prints `sync complete`. Compare the warehouse row count for yesterday against the ledger export count; they must match before you close the alert.
