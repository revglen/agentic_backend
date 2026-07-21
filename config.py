import os
from pathlib import Path

GROQ_API_KEY=os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b") ###"meta-llama/llama-4-scout-17b-16e-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

BASE_DIR=Path(__file__).resolve().parent.parent
VECTOR_INDEX_DIR = os.getenv("VECTOR_INDEX_DIR", str(BASE_DIR / "data" / "vector_index"))
UPLOAD_DIR=os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "society_documents")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
CACHE_RETRIEVER_K = int(os.getenv("CACHE_RETRIEVER_K", "3"))
ENABLE_WEB_CACHE = os.getenv("ENABLE_WEB_CACHE", "true").lower() == "true"

Path(VECTOR_INDEX_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)