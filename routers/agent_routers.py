from pydantic import BaseModel, Field
from typing import List, Literal

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="A question about your society's uploaded documents.",
        examples=["What does our development agreement say about the corpus fund payment timeline?"],
    )


class CompareRequest(BaseModel):
    proposals_text: str = Field(
        default="",
        description=(
            "Free-text proposal details to compare, e.g. pasted from an email or brochure. "
            "Optional if `proposal_labels` is provided instead -- the two are concatenated "
            "when both are given."
        ),
        examples=["Developer A: 1.33x carpet area, 24mo timeline. Developer B: 1.25x, 30mo."],
    )
    use_web_research: bool = Field(
        default=True,
        description="If true, cross-checks the proposals against a live web search (developer track record, typical market terms) before answering.",
    )
    proposal_labels: List[str] = Field(
        default_factory=list,
        description=(
            "Labels (as passed to POST /upload's `label` field) of previously uploaded "
            "proposal documents to pull from the vector store and fold into the comparison."
        ),
        examples=[["developer_a_proposal", "developer_b_proposal"]],
    )


class ComplianceRequest(BaseModel):
    project_details: str = Field(
        default="",
        description=(
            "Society name, ward, developer name, known RERA registration number, etc. "
            "Optional if `doc_labels` is provided instead -- the two are combined when both are given."
        ),
        examples=["Sunshine CHS, Andheri West ward, Developer: ABC Realty"],
    )
    doc_labels: List[str] = Field(
        default_factory=list,
        description=(
            "Labels of previously uploaded supporting documents (RERA certificate, IOD, CC, "
            "NOCs, etc.) to pull from the vector store and fold into the check."
        ),
        examples=[["rera_certificate"]],
    )

class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="A redevelopment or society question. Society documents are checked first; live web search is only used if that's insufficient.",
        examples=["What is the current RERA status of our project, and what does our agreement say about the corpus fund timeline?"],
    )


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Who sent this turn.")
    content: str = Field(..., description="The message text for this turn.")


class ResearchStreamRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="A redevelopment or society question.",
        examples=["What is the current RERA status of our project?"],
    )
    history: List[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns in this conversation, oldest first, used as context for follow-up questions.",
    )


# ---------------------------------------------------------------------------
# Responses (non-streaming)
# ---------------------------------------------------------------------------

class AskResponse(BaseModel):
    answer: str = Field(..., description="The answer, generated from your society documents.")


class CompareResponse(BaseModel):
    comparison: str = Field(..., description="A structured comparison of the developer proposals.")


class ComplianceResponse(BaseModel):
    status_report: str = Field(..., description="RERA/MHADA/municipal approval status summary.")


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The synthesized answer.")
    used_cache: bool = Field(..., description="Whether a cached prior web result was reused.")
    used_live_search: bool = Field(..., description="Whether a fresh live web search was needed.")
    doc_context: str = Field(..., description="Raw retrieved society-document excerpts.")
    web_context: str = Field(..., description="Raw web/cache content actually used, if any.")


class UploadResponse(BaseModel):
    filename: str = Field(..., description="The original filename as uploaded.")
    label: str = Field("", description="The label the document was tagged with, if any.")
    chunks_added: int = Field(..., description="Number of text chunks indexed into the vector store.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(..., description="Always 'ok' if the service is reachable.")


# ---------------------------------------------------------------------------
# Errors
#
# These mirror the JSON shapes actually produced by the global exception
# handlers in main.py, so Swagger's documented error responses match what
# the API really returns instead of FastAPI's default (undocumented) 4xx/5xx.
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error category, e.g. 'http_error'.", examples=["http_error"])
    message: str = Field(..., description="Human-readable explanation of what went wrong.")


class ValidationErrorResponse(BaseModel):
    error: Literal["validation_error"] = "validation_error"
    message: str = Field(..., description="Summary explaining the request body didn't match the expected schema.")
    details: list = Field(..., description="Pydantic's field-by-field validation error list.")


# Reusable `responses=` fragments for router endpoints, keyed by status code.
# These are merged with response_model / status-specific descriptions inline
# in each route decorator so Swagger shows accurate error payloads.
COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Request was missing required content."},
    422: {"model": ValidationErrorResponse, "description": "Request body failed schema validation."},
    502: {"model": ErrorResponse, "description": "The agent failed to produce a result; safe to retry."},
}
