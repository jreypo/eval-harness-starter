# ImagePullBackOff after a deploy

Fix pods stuck in ImagePullBackOff or ErrImagePull after a new release.

## Symptoms

New pods stay in `ImagePullBackOff`. `kubectl describe pod` shows events such as `manifest unknown` or `unauthorized`.

## Diagnose from the event message

`manifest unknown` means the tag does not exist; the pipeline pushed a different tag than the manifest references. `unauthorized` means the imagePullSecret is missing or expired in that namespace. `i/o timeout` points at the registry or the node network.

## Recover

For a wrong tag, roll back the deployment with `kubectl rollout undo deployment/<name>` and fix the pipeline. For expired registry credentials, recreate the pull secret and delete the stuck pods so they retry immediately.
