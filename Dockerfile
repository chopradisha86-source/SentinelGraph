# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files / buffering stdout, so logs show up
# immediately in `docker compose logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements.txt
# changes -- avoids a full reinstall on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code.
COPY src/ ./src/
COPY main.py app.py ./

EXPOSE 8000

# Default: run the live WebSocket server. Override the command to run
# main.py instead, e.g.:
#   docker compose run --rm app python main.py --count 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
