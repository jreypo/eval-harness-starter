# Draining a Kubernetes node for maintenance

Safely evict workloads from a worker node before kernel patching, hardware replacement or instance retirement.

## When to use

Use for planned maintenance on a single node. For a node that is already NotReady and unreachable, skip the drain: cordon it and let the pod eviction timeout reschedule workloads, then follow the provider's instance replacement process.

## Cordon the node

Run `kubectl cordon <node>` so the scheduler stops placing new pods there. Cordoning is reversible and does not touch running pods.

## Drain with PodDisruptionBudgets respected

Run `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data --timeout=10m`. The drain honours PodDisruptionBudgets, so it can block if a deployment has too few healthy replicas elsewhere. Never pass `--disable-eviction` to force past a PDB; scale the affected deployment up first instead.

## Stuck drains

If the drain hangs, list the remaining pods with `kubectl get pods -A --field-selector spec.nodeName=<node>`. A pod blocked by a PDB means the workload lacks spare capacity. A pod stuck in Terminating usually has a finalizer or a hung preStop hook; investigate before deleting it with `--grace-period=0`.

## Return the node to service

After maintenance, confirm the kubelet reports Ready, then run `kubectl uncordon <node>`. Watch for a few minutes that new pods schedule and start normally.
