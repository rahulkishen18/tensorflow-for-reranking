# Use the official TensorFlow image as a parent image
FROM tensorflow/tensorflow:2.15.1

# Set the working directory
WORKDIR /app

# Install Git
RUN apt-get update && \
    apt-get install -y git