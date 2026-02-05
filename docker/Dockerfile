# Use stable Python 3.11 (compatible with Qiskit Aer wheels)
FROM python:3.11-slim

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
# pinning qiskit versions to ensure stability
RUN pip install --no-cache-dir \
    qiskit==1.0.2 \
    qiskit-aer==0.13.3 \
    qiskit-ibm-runtime \
    numpy \
    pandas \
    scipy \
    matplotlib \
    pylatexenc

# Copy the source code into the container
COPY src/ /app/src/
COPY data/ /app/data/

# Default command: run the local quantum test
CMD ["python3", "src/quantum/local_test.py"]
