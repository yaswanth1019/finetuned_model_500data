FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy production application
COPY main.py .

# Environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/transformers

# Create cache directories
RUN mkdir -p /tmp/huggingface /tmp/transformers && \
    chmod 777 /tmp/huggingface /tmp/transformers

# Expose port
EXPOSE 8000

# Extended health check for production model loading
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run production app
CMD ["python", "main.py"]
