# CoreDNS resolution failures

Debug in-cluster DNS failures, which surface as application timeouts that look like the dependency is down.

## Symptoms

Applications log `no such host` or `i/o timeout` when resolving service names. Errors are intermittent and spread across many unrelated services at once.

## Confirm it is DNS

From a debug pod run `nslookup kubernetes.default` and `nslookup <service>.<ns>.svc.cluster.local`. If lookups fail or take seconds while direct IP connections work, the problem is DNS, not the dependency.

## Check CoreDNS health and load

Look at `kubectl -n kube-system get pods -l k8s-app=kube-dns` and their logs. High CPU on CoreDNS pods with many NXDOMAIN answers usually means clients are searching through every search-path suffix; lowering `ndots` in the pod dnsConfig cuts the query volume.

## Scale CoreDNS

If the pods are simply saturated, scale the deployment: `kubectl -n kube-system scale deployment coredns --replicas=<n>`, or tune the dns-autoscaler ConfigMap so replicas follow node count.
