# Ingress TLS certificate rotation

Rotate the TLS certificate served by the shared ingress-nginx controller for a public hostname. Use this when a manually issued certificate is close to expiry or has been compromised. Certificates issued by cert-manager renew automatically; see cert-manager-renewal.md instead.

## When to use

Alert `IngressCertExpiringSoon` fires when a certificate on the ingress expires in less than 14 days. Customer reports of browser TLS warnings on a single hostname also point here. If every hostname is affected at once, check the controller pods first.

## Prerequisites

You need the new certificate chain and private key as PEM files, kubectl access to the application namespace, and the name of the ingress object. Confirm the new certificate covers every hostname in the ingress `tls.hosts` list with `openssl x509 -in tls.crt -noout -text`.

## Create the new secret and patch the ingress

Create a new TLS secret instead of editing the old one, so rollback is a one-line patch: `kubectl -n <ns> create secret tls <host>-tls-2026q4 --cert=tls.crt --key=tls.key`. Then patch the ingress to reference it: `kubectl -n <ns> patch ingress <name> --type=json -p '[{"op":"replace","path":"/spec/tls/0/secretName","value":"<host>-tls-2026q4"}]'`. The controller reloads within about 30 seconds without dropping connections.

## Verify the served certificate

Check what clients actually receive, not what the secret contains: `openssl s_client -connect <host>:443 -servername <host> </dev/null | openssl x509 -noout -dates -subject`. The notAfter date must match the new certificate. Repeat from outside the cluster, because an internal load balancer can serve a cached chain.

## Clean up and rollback

After 24 hours with no TLS errors, delete the old secret with `kubectl -n <ns> delete secret <old-secret>`. To roll back before then, patch the ingress back to the old secret name. Never delete the old secret before verification passes.
