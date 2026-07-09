FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# tzdata für Zeitzone, curl für Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# User Setup und Ordnerstruktur vorab anlegen
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungsdateien direkt mit den richtigen Rechten kopieren
COPY --chown=appuser:appuser . .
COPY --chown=appuser:appuser .env.example .env

# Zu nicht-privilegiertem User wechseln
USER appuser

EXPOSE 8000

# Korrekter Healthcheck auf Port 8000 und der /health Route
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Standard-Fallback, falls man ohne docker-compose startet.
# Wird durch 'command' in docker-compose.yml überschrieben.
CMD ["python", "run.py"]