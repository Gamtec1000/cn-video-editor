FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and ffmpeg with NVIDIA support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create media directories
RUN mkdir -p /media/rendered /media/.temp /media/.jobs

# Environment variables
ENV MEDIA_BASE=/media
ENV PYTHONUNBUFFERED=1

EXPOSE 8004

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
