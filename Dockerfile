# Use the official TensorFlow image as a parent image
FROM google/cloud-sdk:latest

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install mosaicml-cli
RUN pip3 install --upgrade mosaicml-cli

CMD ["python3", "run_mcli.py"]