# Deployment

This directory holds **generic** deployment material: Dockerfiles, a Helm chart and a bootstrap manifest.
Hostnames, node names, tunnel tokens and kubeconfigs are **not** stored here. Cluster runner scale sets
and Cloudflare routing live in separate infrastructure repositories.

```
docker/             api, web (Next.js standalone) and site (nginx static) Dockerfiles
helm/minerva/       chart: api, web, site, PostgreSQL, optional cloudflared, NetworkPolicies, helm test
bootstrap/          namespace + namespace-scoped deployer ServiceAccount/Role (applied by an admin)
```

## Go-live checklist (each live step needs the owner's approval)

1. **CI runners:** add a `minerva-runners` ARC scale set (runner + PostgreSQL sidecar + rootless BuildKit)
   in the cluster repository and install it; make sure the shared GitHub token can access this repository.
2. **Cloudflare:** add an app entry for Minerva in the Cloudflare repository: the site hostname routes to
   `http://<release>-site.<namespace>.svc.cluster.local:80`, the app hostname to
   `http://<release>-web.<namespace>.svc.cluster.local:80`. Merge and apply there; it creates the tunnel.
3. **Bootstrap:** `kubectl apply -f deploy/bootstrap/namespace-and-rbac.yaml` (as an admin).
4. **Secrets (by hand, never committed):**
   ```bash
   kubectl -n minerva create secret generic minerva-secrets \
     --from-literal=postgres-password="$(python3 -c 'import secrets; print(secrets.token_hex(24))')" \
     --from-literal=session-secret="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
   ```
   The tunnel token goes into `cloudflared-token` (key `token`) using the Cloudflare repository's tooling.
   Hex passwords avoid URL-encoding problems in the database URL.
5. **GitHub:** create the `production` environment with secret `KUBE_CONFIG_B64` (the deployer's
   namespace-scoped kubeconfig) and set repository variables `SITE_HOST`, `APP_HOST`, optional
   `HELM_EXTRA_VALUES` (node placement, pull secrets) and finally `DEPLOY_ENABLED=true`.
6. Push to `main`: CI runs, images are published by digest, `helm upgrade --install` runs with
   `--rollback-on-failure`, and `helm test` checks the release.

## Validate locally (safe, read-only)

```bash
D=sha256:$(printf 'a%.0s' {1..64})
helm lint deploy/helm/minerva -f deploy/helm/minerva/values-production.yaml \
  --set api.image.digest=$D --set web.image.digest=$D --set site.image.digest=$D
```

## Notes

- The API is never exposed: a NetworkPolicy allows it to be reached only from the web pods (which proxy
  `/api/v1/*`) and from `helm test`. Rate limits are per process, so the API runs a single replica.
- Registration is open by default (`api.allowRegistration`); set it to `false` for a private instance.
- Putting the app hostname behind Cloudflare Access instead of (or in addition to) Minerva's own accounts is a
  decision for the Cloudflare repository; the chart works either way.
