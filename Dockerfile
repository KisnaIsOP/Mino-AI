# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install gunicorn psutil

# Copy the rest of the application
COPY . .

# Create volume for logs and database
VOLUME ["/app/logs", "/app/data"]

# Expose port
EXPOSE 7007

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7007", "--workers", "4", "--threads", "2", "app:app"]
