# Node disk pressure

Respond to a node reporting the DiskPressure condition, which makes the kubelet evict pods and refuse new ones.

## Symptoms

Pods are evicted with reason `Evicted` and message "The node was low on resource: ephemeral-storage". `kubectl describe node` shows the condition `DiskPressure=True`. Alert `NodeDiskPressure` fires.

## Find what is using the disk

SSH or use a debug pod on the node and run `du -xh --max-depth=2 /var/lib | sort -h | tail`. The usual culprits are container images under `/var/lib/containerd`, large emptyDir volumes, and unrotated container logs under `/var/log/pods`.

## Reclaim space safely

Prune unused images with `crictl rmi --prune`. For a pod writing huge logs, fix the log level or rotation in the application rather than deleting files under `/var/log/pods` by hand, because the kubelet tracks those files. Do not delete anything under `/var/lib/kubelet`.

## Prevent recurrence

Set ephemeral-storage requests and limits on noisy workloads so the scheduler accounts for them. If image churn is the cause, lower the kubelet `imageGCHighThresholdPercent` so garbage collection starts earlier.
