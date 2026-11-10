# Web image: the Vite build served by unprivileged nginx, which also proxies /api to the API.
#   docker build -t doc-processing-web:local apps/web
FROM node:20-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
# The nginx image fills ${API_UPSTREAM} in templates at startup (only variables that are set)
ENV API_UPSTREAM=http://api:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
