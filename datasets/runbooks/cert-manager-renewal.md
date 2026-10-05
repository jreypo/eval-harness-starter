# cert-manager certificate renewal failures

Diagnose a Certificate resource managed by cert-manager that is not renewing.

## Symptoms

The Certificate shows `READY=False` or its `notAfter` is within 7 days and has not moved. Alert `CertManagerCertNotReady` fires.

## Inspect the certificate chain of resources

Run `kubectl describe certificate <name> -n <ns>`, then follow the chain: CertificateRequest, Order, Challenge. The first resource with an error event is where the failure is. `cmctl status certificate <name> -n <ns>` prints the whole chain in one command.

## Common causes

An HTTP-01 challenge fails when the ingress for `/.well-known/acme-challenge/` is not reachable from the internet. A DNS-01 challenge fails when the DNS provider credentials secret has expired. Rate limiting from the ACME server shows up as a 429 in the Order events; wait it out rather than retrying in a loop.

## Force a renewal

After fixing the cause, trigger a renewal with `cmctl renew <name> -n <ns>`. Do not delete the Certificate's secret to force reissue; the ingress would serve no certificate until the new one is issued.
