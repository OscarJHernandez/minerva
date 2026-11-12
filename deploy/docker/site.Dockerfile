# syntax=docker/dockerfile:1
# Build context: repository root. Public URLs are baked into the static export at build time.
FROM node:25-slim AS build
ENV PNPM_HOME=/pnpm PATH=/pnpm:$PATH CI=true
RUN npm install --global pnpm@12.15.0
WORKDIR /repo
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY apps/web/package.json apps/web/package.json
COPY apps/site/package.json apps/site/package.json
COPY packages/ui/package.json packages/ui/package.json
RUN pnpm install --frozen-lockfile --filter @minerva/site...
COPY packages/ui packages/ui
COPY apps/site apps/site
ARG NEXT_PUBLIC_APP_URL=http://localhost:3000
ARG NEXT_PUBLIC_REPO_URL=https://github.com/OscarJHernandez/minerva
ENV NEXT_PUBLIC_APP_URL=$NEXT_PUBLIC_APP_URL NEXT_PUBLIC_REPO_URL=$NEXT_PUBLIC_REPO_URL
RUN pnpm --filter @minerva/site build

FROM nginxinc/nginx-unprivileged:1.31-alpine AS runtime
COPY deploy/docker/site.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /repo/apps/site/out /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
