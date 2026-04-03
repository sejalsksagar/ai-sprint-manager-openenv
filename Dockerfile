FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    pydantic==2.10.3 \
    openai \
    requests

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]