# etcd database size and defragmentation

Reduce the etcd database size when it approaches the space quota, before the cluster goes read-only.

## Symptoms

Alert `EtcdDatabaseQuotaNearlyFull` fires above 80 percent of `--quota-backend-bytes`. If the quota is exceeded, etcd raises the `NOSPACE` alarm and the API server rejects all writes.

## Compact, then defragment one member at a time

Compaction marks old revisions as free; defragmentation returns the space to the filesystem. Run `etcdctl compact <revision>` with the current revision, then `etcdctl defrag --endpoints=<one-member>` on each member in turn, followers first and the leader last. Defragmenting a member blocks it, so never defrag all members at once.

## Clear the alarm

If the NOSPACE alarm was raised, run `etcdctl alarm disarm` after defragmentation. Then confirm writes work with `kubectl create configmap etcd-probe --dry-run=server -o yaml`.
