"""
FastAPI orchestration layer for the Mumbai Redevelopment AI tool.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health

Interactive API docs (Swagger UI): http://localhost:8000/docs
Alternative docs (ReDoc):          http://localhost:8000/redoc
Raw OpenAPI schema:                http://localhost:8000/openapi.json
"""
import os
import numpy as np
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from huggingface_hub import login
import uvicorn
import logging

import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from dotenv import load_dotenv
load_dotenv()

from backend.routers.redevelopment_router import router as redevelopment_router
from backend.routers.ingestion_router import router as ingestion_router
from backend.routers.agent_routers import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mumbai_redevelopment_ai")

try:
    hucking_face_token=os.getenv("HUGGINGFACE_TOKEN")
    login(token=self.hucking_face_token)
    print("✓ Successfully log into Hugging Face Hub...")
except:
    print("Warning: Could not authenticate with HuggingFace Hub")

API_DESCRIPTION = """
Agent backend for Mumbai housing society redevelopment decisions: document
Q&A, hybrid docs+web research, developer proposal comparison, RERA/MHADA/
municipal compliance tracking, and comparable market research.

### How the pieces fit together

* **Ingestion** (`POST /upload`) indexes a PDF/text document into a FAISS
  vector store, optionally tagged with a `label` (e.g. `developer_a_proposal`,
  `rera_certificate`, `society_bylaws`). Any other endpoint that accepts a
  `*_labels` field can later pull that document back out by label.
* **Document Q&A** (`/redevelopment/ask*`) answers questions purely from
  your uploaded society documents.
* **Research** (`/redevelopment/research*`) is the same idea but hybrid: it
  checks your documents first, and only falls back to a live web search (or
  a cached prior one) if your documents aren't enough — and tells you which
  source it used.
* **Compare Proposals** / **Compliance Check** (`/redevelopment/compare-
  proposals*`, `/redevelopment/compliance-check*`) accept either pasted
  text, previously uploaded documents (via labels), or both.
* **Market Research** (`/redevelopment/market-research*`) always runs a live
  web search for comparable rates/deals in a given locality — there's
  nothing of yours to upload for this one.

### Streaming

Every non-streaming endpoint above has a `*_stream` counterpart that
returns a `text/event-stream` (Server-Sent Events) response instead of a
single JSON body. Each line looks like:

```
data: {"type": "token", "content": "partial answer text..."}

data: {"type": "done", "answer": "full answer text", ...extra metadata...}
```

or, on failure mid-stream:

```
data: {"type": "error", "detail": "human-readable message"}
```

Swagger's "Try it out" doesn't render SSE nicely — use `curl -N` against
the `*_stream` URL, or the Streamlit UI, to see it stream token by token.

### Errors

Errors are always JSON with an `error` category and a human-readable
`message` (see `ErrorResponse` / `ValidationErrorResponse` below) — this API
never leaks a raw stack trace to the client.
"""

fast_app = FastAPI(
    title="Mumbai Redevelopment AI",
    description=API_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Mumbai Redevelopment AI",
        "email": "revglen@gmail.com",
    },
    license_info={"name": "Proprietary"},
    openapi_tags=[
        {
            "name": "redevelopment",
            "description": "Document Q&A, hybrid research, proposal comparison, compliance checks, and market research — each with a streaming (`*_stream`) counterpart.",
        },
        {
            "name": "ingestion",
            "description": "Upload PDF/text documents into the vector store, optionally tagged with a label for later retrieval by other endpoints.",
        },
        {
            "name": "health",
            "description": "Service liveness check.",
        },
    ],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 0,   # collapse the schema list at the bottom by default
        "displayRequestDuration": True,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
    },
)


@fast_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Malformed / missing request body fields (Pydantic validation)."""
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "The request body did not match the expected schema.",
            "details": exc.errors(),
        },
    )

@fast_app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles HTTPException raised deliberately anywhere in the app
    (e.g. router endpoints), including 404s for unknown routes."""
    logger.warning("HTTPException on %s %s: %s", request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )

@fast_app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong while processing your request. Please try again.",
        },
)

fast_app.include_router(redevelopment_router)
fast_app.include_router(ingestion_router)

# Allow the Streamlit frontend (different origin) to call this API.
fast_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Streamlit Cloud URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@fast_app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Returns 200 with `{\"status\": \"ok\"}` if the service process is up. Does not check downstream dependencies (LLM provider, vector store on disk, etc).",
    response_model=HealthResponse,
)
def health():
    return {"status": "ok"}

with gr.Blocks() as gradio_ui:
    gr.Markdown("# My FastAPI + Gradio App")
    greet = gr.Interface(fn=lambda name: f"Hello {name}!", inputs="text", outputs="text")

#Mount Gradio on FastAPI
app = gr.mount_gradio_app(fast_app, gradio_ui, path="/")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",   # import string, NOT the app object directly
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
