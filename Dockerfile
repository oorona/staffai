# Use an official Python runtime as a parent image
# Using slim-bullseye for a smaller footprint, adjust Python version as needed (e.g., 3.10, 3.11)
FROM python:3.11-slim-bullseye

# Set environment variables to prevent buffering stdout/stderr
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size by not storing the pip cache
# --upgrade ensures latest versions satisfying requirements are installed
# Consider using virtual environments for better dependency management if needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes bot.py, the cogs/ directory, and the utils/ directory
COPY . .

# Command to run the application when the container launches
# Assumes bot.py is the main entry point
CMD ["python", "main.py"]

# Note: The .env file is NOT copied into the image for security.
# It should be provided to the container at runtime via:
# 1. Docker run -v option: docker run -v $(pwd)/.env:/app/.env ... my-discord-bot
# 2. Docker run --env-file option: docker run --env-file .env ... my-discord-bot
# 3. Docker Compose env_file directive
# 4. Kubernetes Secrets/ConfigMaps