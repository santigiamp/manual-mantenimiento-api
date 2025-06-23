FROM python:3.11-slim

WORKDIR /app

# Solo instalar lo esencial
COPY requirements.txt .

# Instalar dependencias sin cache
RUN pip install --no-cache-dir --no-deps fastapi uvicorn
RUN pip install --no-cache-dir qdrant-client groq

# No instalar torch pesado, usar alternativa
RUN pip install --no-cache-dir sentence-transformers --no-deps
RUN pip install --no-cache-dir transformers tokenizers numpy requests tqdm

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
