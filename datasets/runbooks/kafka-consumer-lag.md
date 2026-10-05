# Kafka consumer lag

Respond when a consumer group falls behind its topic and events are processed late.

## Symptoms

Alert `KafkaConsumerLagHigh` fires when lag exceeds 100k messages for 10 minutes. Downstream systems show delayed updates.

## Find the bottleneck

Run `kafka-consumer-groups --describe --group <group>` and look at per-partition lag. Lag on a single partition points at a hot key or a stuck consumer. Lag spread evenly across partitions means the group lacks throughput.

## Add throughput

Scale the consumer deployment up to at most the number of partitions; consumers beyond the partition count sit idle. If already at the partition count, the topic needs more partitions, which is a change for the owning team.

## Stuck consumer

If one partition is not moving at all, restart the single consumer pod that owns it. Check its logs for a poison message first, because a restart will just hit the same message again.
