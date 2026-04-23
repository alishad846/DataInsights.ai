FROM node:18-bullseye

# Install Python and required ML system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to python3 for child_process spawns
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies globally
COPY ml_engine/requirements.txt ./ml_engine/
RUN pip3 install --no-cache-dir -r ./ml_engine/requirements.txt

# Install Node dependencies
COPY backend-node/package*.json ./backend-node/
WORKDIR /app/backend-node
RUN npm install

# Copy application source code
WORKDIR /app
COPY ml_engine ./ml_engine
COPY backend-node ./backend-node

# Create uploads directory
RUN mkdir -p /app/backend-node/uploads

ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

WORKDIR /app/backend-node
CMD ["npm", "start"]
