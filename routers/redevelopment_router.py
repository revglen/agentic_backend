"""
FastAPI orchestration layer for the Mumbai Redevelopment AI tool.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /redevelopment/ask                     -> RAG Q&A over society documents
    POST /redevelopment/ask_stream               -> streaming version of /ask (SSE)
    POST /redevelopment/compare-proposals        -> compare developer proposals
    POST /redevelopment/compare-proposals_stream -> streaming version (SSE)
    POST /redevelopment/compliance-check         -> RERA/MHADA/approval status lookup
    POST /redevelopment/compliance-check_stream  -> streaming version (SSE)
    POST /redevelopment/research                 -> hybrid docs+web research
    POST /redevelopment/research_stream           -> streaming version (SSE)

See backend/main.py's FastAPI(description=...) for the overall API guide,
and each route decorator below for endpoint-specific docs (all of which
show up in Swagger UI at /docs).
"""
import json
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
import logging

from backend.routers.agent_routers import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    ComplianceRequest,
    ComplianceResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchStreamRequest,
    ErrorResponse,
    COMMON_ERROR_RESPONSES,
)

from backend.agent import Agent
from backend.research_graph import HybridResearchGraph

logger = logging.getLogger("mumbai_redevelopment_ai.redevelopment_router")

router = APIRouter(
    prefix="/redevelopment",
    tags=["redevelopment"],
)

agent = Agent()

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"

def _stream_responses(example: str) -> dict:
    
    return {
        200: {
            "description": (
                "text/event-stream. One `data: {...}` JSON line per event: "
                "`{\"type\": \"token\", \"content\": str}` while generating, "
                "then one final `{\"type\": \"done\", ...}` with the full answer "
                "and metadata, or `{\"type\": \"error\", \"detail\": str}` on failure."
            ),
            "content": {"text/event-stream": {"example": example}},
        },
        **COMMON_ERROR_RESPONSES,
    }


# ---------------------------------------------------------------------------
# Document Q&A
# ---------------------------------------------------------------------------

@router.post(
    "/ask",
    summary="Ask a question about your society documents",
    description=(
        "Answers a question using only your uploaded society documents (no live web search). "
        "Returns 404 if no documents have been uploaded yet."
    ),
    response_model=AskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No society documents are available yet."},
        **COMMON_ERROR_RESPONSES,
    },
)
async def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    try:
        answer = agent.run_doc_qa(req.question)
    except FileNotFoundError as exc:
        logger.error("Document store missing for /ask: %s", exc)
        raise HTTPException(
            status_code=404,
            detail="No society documents are available to answer questions from yet.",
        )
    except Exception as exc:
        logger.exception("run_doc_qa failed for question=%r", req.question)
        raise HTTPException(
            status_code=502,
            detail="The Q&A agent failed to produce an answer. Please try again shortly.",
        )

    return {"answer": answer}


@router.post(
    "/ask_stream",
    summary="Ask a question about your society documents (streaming)",
    description="Streaming (SSE) version of `POST /ask`. The final `done` event additionally carries `doc_context` (the retrieved document excerpts used).",
    responses=_stream_responses(
        'data: {"type": "token", "content": "The corpus fund "}\n\n'
        'data: {"type": "token", "content": "is due within 30 days..."}\n\n'
        'data: {"type": "done", "answer": "The corpus fund is due within 30 days...", "doc_context": "--- bylaws.pdf ---\\n..."}\n\n'
    ),
)
async def ask_stream(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    def event_stream():
        try:
            for event in agent.run_doc_qa_stream(req.question):
                yield _sse(event)
        except Exception:
            logger.exception("run_doc_qa_stream failed for question=%r", req.question)
            yield _sse({
                "type": "error",
                "detail": "The Q&A agent failed to produce an answer. Please try again shortly.",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Compare Proposals
# ---------------------------------------------------------------------------

@router.post(
    "/compare-proposals",
    summary="Compare developer redevelopment proposals",
    description=(
        "Compares proposals supplied as pasted `proposals_text`, previously uploaded "
        "documents referenced by `proposal_labels`, or both (they're concatenated). "
        "Optionally cross-checks against live web research when `use_web_research` is true."
    ),
    response_model=CompareResponse,
    responses=COMMON_ERROR_RESPONSES,
)
async def compare_proposals(req: CompareRequest):
    if not (req.proposals_text or "").strip() and not req.proposal_labels:
        raise HTTPException(
            status_code=400,
            detail="Provide 'proposals_text' and/or at least one 'proposal_labels' entry.",
        )

    try:
        result = agent.run_compare_proposals(req.proposals_text, req.use_web_research, req.proposal_labels)
    except Exception as exc:
        logger.exception("run_compare_proposals failed")
        raise HTTPException(
            status_code=502,
            detail="The proposal comparison agent failed. Please try again shortly.",
        )

    return {"comparison": result}


@router.post(
    "/compare-proposals_stream",
    summary="Compare developer redevelopment proposals (streaming)",
    description=(
        "Streaming (SSE) version of `POST /compare-proposals`. The final `done` event additionally "
        "carries `web_context`, `used_web_research`, and `proposal_labels_used`."
    ),
    responses=_stream_responses(
        'data: {"type": "token", "content": "Developer A offers "}\n\n'
        'data: {"type": "token", "content": "1.33x carpet area over 24 months..."}\n\n'
        'data: {"type": "done", "answer": "Developer A offers 1.33x carpet area over 24 months...", '
        '"web_context": "...", "used_web_research": true, "proposal_labels_used": ["developer_a_proposal", "developer_b_proposal"]}\n\n'
    ),
)
async def compare_proposals_stream(req: CompareRequest):
    if not (req.proposals_text or "").strip() and not req.proposal_labels:
        raise HTTPException(
            status_code=400,
            detail="Provide 'proposals_text' and/or at least one 'proposal_labels' entry.",
        )

    def event_stream():
        try:
            for event in agent.run_compare_proposals_stream(
                req.proposals_text, req.use_web_research, req.proposal_labels
            ):
                yield _sse(event)
        except Exception:
            logger.exception("run_compare_proposals_stream failed")
            yield _sse({
                "type": "error",
                "detail": "The proposal comparison agent failed. Please try again shortly.",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Compliance Check
# ---------------------------------------------------------------------------

@router.post(
    "/compliance-check",
    summary="Check RERA / MHADA / municipal compliance status",
    description=(
        "Looks up current approval status via live web search, optionally enriched with "
        "previously uploaded supporting documents referenced by `doc_labels` (RERA certificate, "
        "IOD, CC, NOCs, etc.)."
    ),
    response_model=ComplianceResponse,
    responses=COMMON_ERROR_RESPONSES,
)
async def compliance_check(req: ComplianceRequest):
    if not (req.project_details or "").strip() and not req.doc_labels:
        raise HTTPException(
            status_code=400,
            detail="Provide 'project_details' and/or at least one 'doc_labels' entry.",
        )

    try:
        result = agent.run_compliance_check(req.project_details, req.doc_labels)
    except Exception as exc:
        logger.exception("run_compliance_check failed")
        raise HTTPException(
            status_code=502,
            detail="The compliance check agent failed. Please try again shortly.",
        )

    return {"status_report": result}


@router.post(
    "/compliance-check_stream",
    summary="Check RERA / MHADA / municipal compliance status (streaming)",
    description=(
        "Streaming (SSE) version of `POST /compliance-check`. The final `done` event additionally "
        "carries `search_results` and `doc_labels_used`."
    ),
    responses=_stream_responses(
        'data: {"type": "token", "content": "RERA registration "}\n\n'
        'data: {"type": "token", "content": "P51800012345 is active through..."}\n\n'
        'data: {"type": "done", "answer": "RERA registration P51800012345 is active through...", '
        '"search_results": "...", "doc_labels_used": ["rera_certificate"]}\n\n'
    ),
)
async def compliance_check_stream(req: ComplianceRequest):
    if not (req.project_details or "").strip() and not req.doc_labels:
        raise HTTPException(
            status_code=400,
            detail="Provide 'project_details' and/or at least one 'doc_labels' entry.",
        )

    def event_stream():
        try:
            for event in agent.run_compliance_check_stream(req.project_details, req.doc_labels):
                yield _sse(event)
        except Exception:
            logger.exception("run_compliance_check_stream failed")
            yield _sse({
                "type": "error",
                "detail": "The compliance check agent failed. Please try again shortly.",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Hybrid Research (docs first, web if needed, cached)
# ---------------------------------------------------------------------------

@router.post(
    "/research",
    summary="Ask a question, checking documents first and the live web only if needed",
    description=(
        "Retrieves from your society documents; if that's graded insufficient, checks a cache of "
        "prior web searches; if that's also insufficient, runs a fresh live web search and caches "
        "it for next time. The response tells you which source(s) were actually used."
    ),
    response_model=ResearchResponse,
    responses=COMMON_ERROR_RESPONSES,
)
async def research(req: ResearchRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    try:
        research_graph = HybridResearchGraph()
        result = research_graph.run_research(req.question)
    except Exception as exc:
        logger.exception("research failed")
        raise HTTPException(
            status_code=502,
            detail="The research agent failed. Please try again shortly.",
        )

    return result

@router.post(
    "/research_stream",
    summary="Ask a question, checking documents first and the live web only if needed (streaming)",
    description=(
        "Streaming (SSE) version of `POST /research`. Accepts a `history` array of prior chat turns "
        "for follow-up questions. The final `done` event carries `used_cache`, `used_live_search`, "
        "`doc_context`, and `web_context` in addition to the full `answer`."
    ),
    responses=_stream_responses(
        'data: {"type": "token", "content": "Based on your bylaws, "}\n\n'
        'data: {"type": "token", "content": "the corpus fund is due 30 days after..."}\n\n'
        'data: {"type": "done", "answer": "Based on your bylaws, the corpus fund is due 30 days after...", '
        '"used_cache": false, "used_live_search": false, "doc_context": "...", "web_context": ""}\n\n'
    ),
)
async def research_stream(req: ResearchStreamRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    def event_stream():
        try:
            research_graph = HybridResearchGraph()
            history = [h.model_dump() for h in req.history]
            for event in research_graph.run_research_stream(req.question, history):
                yield _sse(event)
        except Exception:
            logger.exception("research streaming failed")
            yield _sse({
                "type": "error",
                "detail": "The research agent failed. Please try again shortly.",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
