FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ACCORD_HOST=0.0.0.0
ENV ACCORD_PORT=7878

EXPOSE 7878

HEALTHCHECK --interval=5s --timeout=3s --retries=12 \
    CMD curl -sf http://localhost:7878/api/health || exit 1

CMD ["python", "-m", "orchestrator.main"]
