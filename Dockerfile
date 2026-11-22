# Offline, dependency-free image. Base image pinned by digest (python 3.11.x, Debian slim).
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY . /app

# No pip install: the project uses the standard library only.
RUN useradd --create-home --uid 10001 auditor && chown -R auditor:auditor /app
USER auditor

# Single verification command (run with --network none).
CMD ["python", "-m", "auditenv", "check"]
