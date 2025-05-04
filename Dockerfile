FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories with proper permissions
RUN mkdir -p /app/data/vector_db && chmod 777 /app/data/vector_db

COPY requirements.txt ./
COPY app.py ./
COPY src/ ./src/
COPY utils/ ./utils/
# Don't copy data folder, let it be created at runtime
# COPY data/ ./data/

RUN pip3 install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]