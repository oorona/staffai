# 1. Base Image (Choose an appropriate Python version)
FROM python:3.11-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# 4. Install System Dependencies (if any, e.g., for libraries that need them)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # any-system-dependencies-here \
#     && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies (including spacy)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Download SpaCy Models (COMMENTED OUT - spacy not in requirements.txt)
# These lines are commented out because spacy is not currently in requirements.txt
# Uncomment if you add spacy to requirements.txt
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download es_core_news_sm
# If you decide to make model names configurable via build arguments,
# you could use ARGs here, e.g.:
# ARG SPACY_EN_MODEL=en_core_web_sm
# ARG SPACY_ES_MODEL=es_core_news_sm
# RUN python -m spacy download ${SPACY_EN_MODEL}
# RUN python -m spacy download ${SPACY_ES_MODEL}

# 7. Copy the rest of your application code
COPY . /app/

# 8. (Optional) Expose Port if your app is a web server (not typical for a Discord bot directly)
# EXPOSE 3000 

# 9. Command to run your application
CMD ["python", "main.py"]