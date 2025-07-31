# Multi-stage build for EvalSync
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Newman (Postman CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g newman newman-reporter-htmlextra

# Create non-root user
RUN groupadd -r evalsync && useradd -r -g evalsync evalsync

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY collections/ ./collections/
COPY docker/entrypoint.sh ./entrypoint.sh

# Create directories for reports and test data
RUN mkdir -p /app/reports /app/test_data /app/logs && \
    chown -R evalsync:evalsync /app

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Switch to non-root user
USER evalsync

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTEST_CACHE_DIR=/app/.pytest_cache

# Expose port for test results server
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command (run test suite)
CMD ["test"]