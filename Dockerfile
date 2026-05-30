# Use an official lightweight Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies if any are needed (minimal slim image setup)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml to install project dependencies first (improves Docker layer caching)
COPY pyproject.toml /app/

# Install dependencies (in non-editable, production-ready form)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[all]

# Copy the rest of the application codebase
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY data/ /app/data/
COPY scripts/ /app/scripts/
COPY .env.example /app/

# Establish default environmental pathways
ENV GEMINI_API_KEY=""
ENV SLACK_WEBHOOK_URL=""
ENV MRD_DATA_DIR="/app/data"
ENV MRD_PROMPTS_DIR="/app/prompts"

# Ensure run script is executable
RUN chmod +x /app/scripts/run_eval.py

# Define entrypoint to allow container to behave as an executable CLI
ENTRYPOINT ["python", "scripts/run_eval.py"]

# Default command options (can be overridden when running the container)
CMD ["--concurrency", "2"]
