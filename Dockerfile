FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# CRITICAL FIX: Install the CPU-only version of PyTorch first. 
# The default PyTorch includes massive CUDA GPU binaries (500MB+) that cause 
# "Bus error (core dumped)" and "Read-only file system" errors in Docker Desktop 
# due to running out of RAM/Disk space during extraction.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
