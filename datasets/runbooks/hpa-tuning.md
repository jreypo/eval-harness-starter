# Handling a traffic spike with the HorizontalPodAutoscaler

Respond when a deployment is saturated by traffic faster than the autoscaler reacts, or when the HPA is pinned at its maximum.

## Symptoms

Latency rises and the HPA shows `current replicas = maxReplicas`. `kubectl get hpa <name>` reports CPU well above target.

## Immediate mitigation

Raise the ceiling rather than scaling the deployment directly, because the HPA will scale a manually edited deployment back down: `kubectl patch hpa <name> -p '{"spec":{"maxReplicas":<n>}}'`. Check the cluster has node capacity for the new replicas, or the pods will sit Pending.

## Tune the scale-up behaviour

For spiky traffic, set `behavior.scaleUp.stabilizationWindowSeconds` to 0 and allow a large percentage step so the HPA reacts within one sync period. Keep a scale-down stabilization window of several minutes to avoid flapping.

## After the spike

Review whether the new maximum should stay. Record the peak request rate so the capacity plan reflects it.
