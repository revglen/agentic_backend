"""
FastAPI orchestration layer for the Mumbai Redevelopment AI tool.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /upload             -> ingest a document into the vector store
"""
import shutil
from pathlib import Path
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from backend.vectorstore import VectorStore
from backend import config
from backend.routers.agent_routers import UploadResponse, ErrorResponse

logger = logging.getLogger("mumbai_redevelopment_ai.ingestion_router")

router = APIRouter(tags=["ingestion"])

# Adjust to whatever your ingestion pipeline actually supports.
ALLOWED_SUFFIXES = {".pdf", ".txt", ".docx", ".md"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

@router.post(
    "/upload",
    summary="Upload and index a document",
    description=(
        "Uploads a PDF/text/docx/md file, splits it into chunks, and indexes it into the society-"
        "documents vector store. The optional `label` (form field, not JSON) tags every chunk so "
        "it can later be retrieved specifically -- e.g. `developer_a_proposal`, `rera_certificate`, "
        f"`society_bylaws`. Max file size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB; allowed types: "
        f"{', '.join(sorted(ALLOWED_SUFFIXES))}."
    ),
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing filename or the file was empty."},
        413: {"model": ErrorResponse, "description": "File exceeded the upload size limit."},
        415: {"model": ErrorResponse, "description": "Unsupported file type."},
        422: {"model": ErrorResponse, "description": "File contents could not be parsed/ingested (corrupt, unreadable, or empty extracted text)."},
        500: {"model": ErrorResponse, "description": "Server-side storage or ingestion failure; safe to retry."},
        502: {"model": ErrorResponse, "description": "Unexpected failure ingesting the file into the vector store; safe to retry."},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="The PDF/text/docx/md file to upload."),
    label: str = Form(
        "",
        description="Optional tag for this document (e.g. 'developer_a_proposal', 'rera_certificate', 'society_bylaws'). Used by other endpoints' *_labels fields to retrieve this document specifically.",
    ),
):
    """Upload a PDF or text file and ingest it into the vector store."""

    # --- 1. Validate the incoming file before touching disk -----------------
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    original_name = Path(file.filename).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. "
                   f"Allowed types: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )

    try:
        upload_dir = Path(config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.exception("Could not create/access upload directory %s", config.UPLOAD_DIR)
        raise HTTPException(status_code=500, detail="Server storage is unavailable. Please try again later.")

    dest = upload_dir / f"{uuid.uuid4().hex}_{original_name}"

    # --- 2. Save the upload to disk, enforcing a size limit -----------------
    bytes_written = 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        logger.exception("Failed to write uploaded file to disk: %s", dest)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file. Please try again.")
    finally:
        await file.close()

    if bytes_written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- 3. Ingest into the vector store ------------------------------------
    try:
        vectorstore: VectorStore = VectorStore()
        n_chunks = vectorstore.ingest_file(str(dest), doc_label=label)
    except FileNotFoundError as exc:
        logger.error("Ingestion could not find saved file %s: %s", dest, exc)
        raise HTTPException(status_code=500, detail="Uploaded file could not be located for ingestion.")
    except ValueError as exc:
        # e.g. unreadable / corrupt / empty-content file the parser rejected
        logger.warning("Ingestion rejected file %s: %s", dest, exc)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not process file contents: {exc}")
    except Exception as exc:
        logger.exception("Unexpected failure ingesting file %s", dest)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail="Failed to ingest file into the vector store.")

    if not n_chunks:
        logger.warning("Ingestion produced 0 chunks for file %s", dest)
        raise HTTPException(
            status_code=422,
            detail="No content could be extracted from this file.",
        )

    return {"filename": original_name, "label": label, "chunks_added": n_chunks}
