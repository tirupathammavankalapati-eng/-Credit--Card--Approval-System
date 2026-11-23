FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (for matplotlib and scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for data and models
RUN mkdir -p data models instance

# Train the model during build (optional, or it will auto-train on first request)
# RUN python scripts/train_model.py

EXPOSE 5000

CMD ["python", "run.py"]
