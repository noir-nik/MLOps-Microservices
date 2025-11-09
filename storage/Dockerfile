FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY app.py .

RUN mkdir -p logs data/datasets data/models data/results db

EXPOSE 8002

CMD ["python", "app.py"]
