# Use specific Python version as requested
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g. build tools)
# For sentencepiece and torch, we might need some basic libs, but slim usually works for pure python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the cache directory so it's ready for the app or volume
RUN mkdir -p /app/model_cache /app/reports

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Pre-load the model (Optional but professional)
RUN python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
    T5Tokenizer.from_pretrained('t5-small', cache_dir='./model_cache'); \
    T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='./model_cache')"

# Expose port
ENV FLASK_APP=src/app.py
EXPOSE 5000

# Run the application
CMD ["python", "-m", "src.app"]
