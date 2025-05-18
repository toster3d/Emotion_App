# Build stage
FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage (serve static files using node)
FROM node:18-alpine

WORKDIR /app

COPY --from=build /app/build ./build

# Installation of the static file server (serve)
RUN npm install -g serve

EXPOSE 3001

CMD ["serve", "-s", "build", "-l", "3001"]
