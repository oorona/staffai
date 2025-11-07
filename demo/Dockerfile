FROM python:3.13-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir openai python-dotenv fastmcp aiohttp httpx

# Copy files from demo folder (self-contained)
COPY validate_system_prompt.py .
COPY test_tool_calling.py .
COPY personality_prompt.txt .
COPY response_schema.json .
COPY .env .

CMD ["python3", "-u", "validate_system_prompt.py"]
