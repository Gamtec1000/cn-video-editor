FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ffmpeg and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create media directories
RUN mkdir -p /media/scenes /media/.jobs

# Environment variables
ENV MEDIA_BASE=/media
ENV PYTHONUNBUFFERED=1

EXPOSE 8006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
