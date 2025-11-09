FROM node:24-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps || npm i
COPY . .
RUN npm run build --omit=dev
# RUN npm prune --omit=dev

FROM node:24-alpine AS runtime
WORKDIR /app
COPY --from=builder /app/build build/
COPY --from=builder /app/node_modules node_modules/
COPY --from=builder /app/package.json .
COPY --from=builder /app/package-lock.json .

EXPOSE 3000
ENV NODE_ENV=production
# ENV VITE_WEB_MASTER_API=$API_URL
CMD [ "node", "build"]

