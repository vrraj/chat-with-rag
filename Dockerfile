# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies first
# This improves layer caching performance. This is the line that executes
# when you build the image, installing all dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This assumes your structure is /backend, start.py, requirements.txt, etc.
COPY . /app

# Expose the port used by Uvicorn
EXPOSE 8000
