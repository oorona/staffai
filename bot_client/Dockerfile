FROM python:3.11-slim

WORKDIR /app

# Copy parent requirements first (bot dependencies)
# Note: context is parent directory, so requirements.txt is at root of context
COPY requirements.txt /app/parent_requirements.txt

# Copy test tool requirements
COPY bot_client/requirements.txt /app/test_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/parent_requirements.txt && \
    pip install --no-cache-dir -r /app/test_requirements.txt

# Copy parent modules (bot code)
COPY *.py /app/
COPY *.json /app/
COPY utils /app/utils
COPY cogs /app/cogs

# Create test client directory
RUN mkdir -p /app/bot_client/prompts

# Copy test tool files
COPY bot_client/bot_api_test.py /app/bot_client/
COPY bot_client/run_automated_test.py /app/bot_client/
COPY bot_client/prompts/test_prompts.txt /app/bot_client/prompts/test_prompts.txt

WORKDIR /app/bot_client

# Run interactively
CMD ["python", "-u", "bot_api_test.py"]
