# ── Stage 1: Builder ──
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# gcc und git für Kompilierung und git-basierte pip-Pipelines (tvdatafeed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y pip setuptools wheel

# ── Stage 2: Runtime ──
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Sicherheits-Upgrades + tzdata & curl (für Healthcheck), verwundbare System-Pakete entfernen
RUN apt-get update && apt-get dist-upgrade -y \
    && apt-get install -y --no-install-recommends \
       curl \
       tzdata \
    && apt-get purge -y --auto-remove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip* \
              /usr/local/lib/python3.12/site-packages/setuptools* \
              /usr/local/lib/python3.12/site-packages/wheel* \
              /usr/local/lib/python3.12/site-packages/pkg_resources \
              /usr/local/bin/pip*

# Venv aus Builder kopieren
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Nicht-privilegierten User anlegen & Ordnerstruktur erstellen
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

# Anwendungsdateien kopieren
COPY --chown=appuser:appuser . .
COPY --chown=appuser:appuser .env.example .env

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py"]