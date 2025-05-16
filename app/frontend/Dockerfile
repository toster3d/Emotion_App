# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Development stage
FROM node:18-alpine AS development

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Set environment variables
ENV NODE_ENV=development
ENV PORT=3000

# The source code will be mounted as a volume
# Command for running the development server is specified in docker-compose.yml
CMD ["npm", "start"]

# Production stage
FROM nginx:alpine AS production

# Copy built files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Add custom nginx config if needed
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 