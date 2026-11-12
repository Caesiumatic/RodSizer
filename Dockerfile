FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 rodsizer \
    && mkdir -p /app/uploads /app/results \
    && chown -R rodsizer:rodsizer /app
COPY --chown=rodsizer:rodsizer backend/ ./backend/
COPY --chown=rodsizer:rodsizer frontend/ ./frontend/

USER rodsizer
ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
