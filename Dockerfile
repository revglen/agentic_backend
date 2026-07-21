FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend
COPY . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
EXPOSE 7860

# Adjust "main:app" if your FastAPI entrypoint file/variable is named differently
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
