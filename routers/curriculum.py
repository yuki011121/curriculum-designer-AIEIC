"""
Curriculum Designer routes — INTERFACE_CONTRACT.md §458–554.

Stage B: LLM generation via LangGraph + Azure OpenAI.
Adds upload-material and upload-instructions endpoints for RAG context.
"""

from __future__ import annotations

import io
import json
import html
from datetime import datetime, timezone

import pypdf
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from models.curriculum import (
    ApproveRequest,
    CheckTyposResponse,
    FeedbackEntry,
    GenerateRequest,
    LabMaterial,
    RequestChangesRequest,
    UploadAckResponse,
    UploadInstructionsRequest,
)

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


# ── Error helpers ─────────────────────────────────────────────────────────────

def _not_found(lab_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "LAB_NOT_FOUND",
                "message": f"No lab with id '{lab_id}'",
                "agent": "curriculum-designer",
            }
        },
    )


# ── PDF extraction helper ─────────────────────────────────────────────────────

async def _extract_pdf_text(file: UploadFile) -> str:
    data = await file.read()
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "PDF_NO_TEXT",
                    "message": (
                        "PDF appears to be image-only or empty. "
                        "No text could be extracted."
                    ),
                    "agent": "curriculum-designer",
                }
            },
        )
    return text


# ── PDF export helpers ────────────────────────────────────────────────────────

def _safe_filename_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "lab"


def _markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    try:
        import markdown as md
        from xhtml2pdf import pisa
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "PDF_EXPORT_DEPENDENCY_MISSING",
                    "message": (
                        "PDF export requires 'markdown' and 'xhtml2pdf'. "
                        "Install dependencies and retry."
                    ),
                    "agent": "curriculum-designer",
                }
            },
        ) from exc

    body_html = md.markdown(
        markdown_text,
        extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
    )
    safe_title = html.escape(title)
    html_doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{safe_title}</title>
    <style>
      @page {{
        size: A4;
        margin: 20mm 16mm;
      }}
      body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11pt;
        color: #1f2328;
        line-height: 1.55;
      }}
      h1 {{
        font-size: 24pt;
        margin: 0 0 10pt 0;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 6pt;
      }}
      h2 {{
        font-size: 16pt;
        margin: 18pt 0 8pt 0;
      }}
      h3 {{
        font-size: 13pt;
        margin: 14pt 0 6pt 0;
      }}
      p {{
        margin: 0 0 8pt 0;
      }}
      ul, ol {{
        margin: 0 0 8pt 18pt;
      }}
      li {{
        margin: 0 0 4pt 0;
      }}
      code {{
        font-family: Courier, monospace;
        background: #f6f8fa;
      }}
      pre {{
        font-family: Courier, monospace;
        font-size: 10pt;
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        padding: 8pt;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin: 6pt 0 12pt 0;
      }}
      th, td {{
        border: 1px solid #d0d7de;
        padding: 6pt;
        text-align: left;
        vertical-align: top;
      }}
      th {{
        background: #f6f8fa;
      }}
      hr {{
        border: 0;
        border-top: 1px solid #d0d7de;
        margin: 12pt 0;
      }}
    </style>
  </head>
  <body>{body_html}</body>
</html>
"""
    output = io.BytesIO()
    status = pisa.CreatePDF(html_doc, dest=output, encoding="utf-8")
    if status.err:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "PDF_EXPORT_FAILED",
                    "message": "Unable to render PDF from markdown content.",
                    "agent": "curriculum-designer",
                }
            },
        )
    return output.getvalue()


def _pdf_download_response(lab_id: str, kind: str, title: str, markdown_text: str) -> Response:
    safe_lab_id = _safe_filename_component(lab_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{safe_lab_id}_{kind}_{stamp}.pdf"
    return Response(
        content=_markdown_to_pdf_bytes(title=title, markdown_text=markdown_text),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _lab_export_markdown(material: LabMaterial) -> str:
    objective_lines = [f"- {objective}" for objective in material.learning_objectives]
    return "\n".join(
        [
            f"# {material.title}",
            "",
            "## Metadata",
            f"- **Lab ID:** `{material.lab_id}`",
            f"- **Course ID:** `{material.course_id}`",
            f"- **Version:** `{material.version}`",
            f"- **Difficulty:** `{material.difficulty}`",
            f"- **Estimated Duration:** `{material.estimated_duration_min}` minutes",
            "",
            "## Learning Objectives",
            *objective_lines,
            "",
            "## Lab Spec",
            material.spec_markdown,
        ]
    )


def _quiz_export_markdown(material: LabMaterial) -> str:
    lines = [
        f"# Quiz - {material.title}",
        "",
        "## Metadata",
        f"- **Lab ID:** `{material.lab_id}`",
        f"- **Version:** `{material.version}`",
        "",
    ]
    for question in material.quiz:
        lines.extend(
            [
                f"## {question.id}",
                f"- **Type:** `{question.type}`",
                f"- **Difficulty:** `{question.difficulty}`",
                f"- **Points:** `{question.rubric_points}`",
                (
                    f"- **Learning Objective:** `{question.learning_objective_ref}`"
                    if question.learning_objective_ref
                    else "- **Learning Objective:** _Not specified_"
                ),
                "",
                "### Question",
                question.question,
                "",
                "### Expected Answer",
                question.expected_answer,
                "",
            ]
        )
    return "\n".join(lines)


def _rubric_export_markdown(material: LabMaterial) -> str:
    rubric = material.rubric
    lines = [
        f"# Rubric - {material.title}",
        "",
        "## Metadata",
        f"- **Lab ID:** `{material.lab_id}`",
        f"- **Version:** `{material.version}`",
        "",
        "## Component Weights",
        "| Component | Weight |",
        "| --- | ---: |",
        f"| Code | {rubric.code_weight * 100:.0f}% |",
        f"| Report | {rubric.report_weight * 100:.0f}% |",
        f"| Manual | {rubric.manual_weight * 100:.0f}% |",
        "",
        "## Criteria",
    ]
    for idx, criterion in enumerate(rubric.criteria, start=1):
        lines.extend(
            [
                f"### {idx}. {criterion.name}",
                f"- **Weight:** {criterion.weight * 100:.0f}%",
                criterion.description,
                "",
            ]
        )
    return "\n".join(lines)


# ── Graph invocation helper ───────────────────────────────────────────────────

async def _run_graph(
    request: Request,
    req: GenerateRequest,
    material_content: str | None,
    agent_instructions: str | None,
    feedback: str | None,
    existing_spec: str | None,
) -> dict:
    graph = request.app.state.generation_graph
    llm_client = request.app.state.llm_client
    initial_state = {
        "request": req,
        "material_content": material_content,
        "agent_instructions": agent_instructions,
        "feedback": feedback,
        "spec_markdown": existing_spec,
        "quiz": None,
        "rubric": None,
        "self_review_notes": [],
        "retry_count": 0,
        "llm_client": llm_client,
    }
    return await graph.ainvoke(initial_state)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=LabMaterial)
async def generate(req: GenerateRequest, request: Request) -> LabMaterial:
    """Generate a new lab from learning objectives.

    If a lab already exists with this lab_id, any previously uploaded
    material_content and agent_instructions are preserved and used as
    LLM context. The spec/quiz/rubric are fully regenerated.
    """
    store = request.app.state.store

    # Preserve RAG context from a prior upload on the same lab_id
    existing = store.get(req.lab_id)
    material_content = existing.material_content if existing else None
    agent_instructions = existing.agent_instructions if existing else None

    now = datetime.now(timezone.utc)
    final_state = await _run_graph(
        request, req,
        material_content=material_content,
        agent_instructions=agent_instructions,
        feedback=None,
        existing_spec=None,
    )

    material = LabMaterial(
        lab_id=req.lab_id,
        course_id=req.course_id,
        title=req.title,
        spec_markdown=final_state["spec_markdown"],
        quiz=final_state["quiz"],
        rubric=final_state["rubric"],
        learning_objectives=req.learning_objectives,
        difficulty=req.difficulty,
        estimated_duration_min=req.estimated_duration_min,
        material_content=material_content,
        agent_instructions=agent_instructions,
        approval_status="pending",
        version=1,
        generated_at=now,
        last_updated=now,
    )
    store.put(material)
    return material


@router.post("/generate-with-material", response_model=LabMaterial)
async def generate_with_material(
    request: Request,
    lab_id: str = Form(...),
    title: str = Form(...),
    learning_objectives: str = Form(...),  # JSON-encoded list e.g. '["obj1","obj2"]'
    difficulty: str = Form("intermediate"),
    estimated_duration_min: int = Form(90),
    instructor_id: str = Form(...),
    course_id: str = Form("csc580"),
    file: UploadFile | None = File(None),
) -> LabMaterial:
    """One-shot endpoint: upload PDF + form fields → generate LabMaterial in a single request.

    If no PDF is uploaded, any previously stored material_content for this lab_id is preserved.
    learning_objectives must be a JSON-encoded list string.
    """
    store = request.app.state.store

    # Extract PDF text if a real file was uploaded
    material_content: str | None = None
    if file and file.filename:
        content_type = file.content_type or ""
        if content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=415,
                detail={
                    "error": {
                        "code": "UNSUPPORTED_MEDIA_TYPE",
                        "message": "Only PDF files are accepted.",
                        "agent": "curriculum-designer",
                    }
                },
            )
        material_content = await _extract_pdf_text(file)

    # Preserve existing context when no new PDF is provided
    existing = store.get(lab_id)
    if material_content is None and existing:
        material_content = existing.material_content
    agent_instructions = existing.agent_instructions if existing else None

    # Parse learning objectives — accept JSON list or newline-separated plain text
    try:
        objectives: list[str] = json.loads(learning_objectives)
    except (json.JSONDecodeError, ValueError):
        objectives = [o.strip() for o in learning_objectives.splitlines() if o.strip()]

    req = GenerateRequest(
        lab_id=lab_id,
        course_id=course_id,
        title=title,
        learning_objectives=objectives,
        difficulty=difficulty,
        estimated_duration_min=estimated_duration_min,
        instructor_id=instructor_id,
    )

    now = datetime.now(timezone.utc)
    final_state = await _run_graph(
        request, req,
        material_content=material_content,
        agent_instructions=agent_instructions,
        feedback=None,
        existing_spec=None,
    )

    material = LabMaterial(
        lab_id=lab_id,
        course_id=course_id,
        title=title,
        spec_markdown=final_state["spec_markdown"],
        quiz=final_state["quiz"],
        rubric=final_state["rubric"],
        learning_objectives=objectives,
        difficulty=difficulty,
        estimated_duration_min=estimated_duration_min,
        material_content=material_content,
        agent_instructions=agent_instructions,
        approval_status="pending",
        version=1,
        generated_at=now,
        last_updated=now,
    )
    store.put(material)
    return material


@router.get("/{lab_id}", response_model=LabMaterial)
async def get_lab(lab_id: str, request: Request) -> LabMaterial:
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)
    return material


@router.get("/{lab_id}/export/lab.pdf")
async def export_lab_pdf(lab_id: str, request: Request) -> Response:
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    return _pdf_download_response(
        lab_id=lab_id,
        kind="lab",
        title=f"Lab Spec - {material.title}",
        markdown_text=_lab_export_markdown(material),
    )


@router.get("/{lab_id}/export/quiz.pdf")
async def export_quiz_pdf(lab_id: str, request: Request) -> Response:
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    return _pdf_download_response(
        lab_id=lab_id,
        kind="quiz",
        title=f"Quiz - {material.title}",
        markdown_text=_quiz_export_markdown(material),
    )


@router.get("/{lab_id}/export/rubric.pdf")
async def export_rubric_pdf(lab_id: str, request: Request) -> Response:
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    return _pdf_download_response(
        lab_id=lab_id,
        kind="rubric",
        title=f"Rubric - {material.title}",
        markdown_text=_rubric_export_markdown(material),
    )


@router.post("/{lab_id}/upload-material", response_model=UploadAckResponse)
async def upload_material(
    lab_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> UploadAckResponse:
    """Upload a PDF to use as RAG context for generation.

    The lab must already exist (created via POST /curriculum/generate).
    Extracted text is stored on the LabMaterial and included verbatim
    in all subsequent LLM prompts for this lab.
    """
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    content_type = file.content_type or ""
    if content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=415,
            detail={
                "error": {
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "Only PDF files are accepted.",
                    "agent": "curriculum-designer",
                }
            },
        )

    text = await _extract_pdf_text(file)
    material.material_content = text
    material.last_updated = datetime.now(timezone.utc)
    store.put(material)

    return UploadAckResponse(
        lab_id=lab_id,
        field="material_content",
        chars_stored=len(text),
        message="PDF text extracted and stored. Re-generate to apply.",
    )


@router.post("/{lab_id}/upload-instructions", response_model=UploadAckResponse)
async def upload_instructions(
    lab_id: str,
    body: UploadInstructionsRequest,
    request: Request,
) -> UploadAckResponse:
    """Store freeform instructor instructions used as additional LLM context."""
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    material.agent_instructions = body.instructions
    material.last_updated = datetime.now(timezone.utc)
    store.put(material)

    return UploadAckResponse(
        lab_id=lab_id,
        field="agent_instructions",
        chars_stored=len(body.instructions),
        message="Agent instructions stored. Re-generate to apply.",
    )


@router.post("/{lab_id}/approve", response_model=LabMaterial)
async def approve(lab_id: str, body: ApproveRequest, request: Request) -> LabMaterial:
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    material.approval_status = "approved"
    material.approved_by = body.approved_by
    material.approval_notes = body.notes
    material.last_updated = datetime.now(timezone.utc)
    store.put(material)
    return material


@router.post("/{lab_id}/request-changes", response_model=LabMaterial)
async def request_changes(
    lab_id: str, body: RequestChangesRequest, request: Request,
) -> LabMaterial:
    """Record instructor feedback and regenerate quiz + rubric.

    The spec is preserved (spec-skip pattern in spec_generator_node).
    Only the quiz and rubric are regenerated with the feedback injected.
    """
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    now = datetime.now(timezone.utc)

    # Reconstruct a GenerateRequest from the stored material
    req = GenerateRequest(
        lab_id=material.lab_id,
        course_id=material.course_id,
        title=material.title,
        learning_objectives=material.learning_objectives,
        difficulty=material.difficulty,
        estimated_duration_min=material.estimated_duration_min,
        instructor_id=body.requested_by,
    )

    final_state = await _run_graph(
        request, req,
        material_content=material.material_content,
        agent_instructions=material.agent_instructions,
        feedback=body.feedback,
        existing_spec=material.spec_markdown,   # spec preserved; quiz/rubric regenerated
    )

    new_version = material.version + 1
    material.quiz = final_state["quiz"]
    material.rubric = final_state["rubric"]
    material.version = new_version
    material.approval_status = "needs_changes"
    material.approved_by = None
    material.approval_notes = None
    material.last_updated = now
    material.feedback_history.append(
        FeedbackEntry(
            feedback=body.feedback,
            requested_by=body.requested_by,
            requested_at=now,
            resulting_version=new_version,
        )
    )
    store.put(material)
    return material


@router.post("/{lab_id}/check-typos", response_model=CheckTyposResponse)
async def check_typos(lab_id: str, request: Request) -> CheckTyposResponse:
    """Run LLM proofreading across spec, quiz, and rubric."""
    store = request.app.state.store
    material = store.get(lab_id)
    if material is None:
        raise _not_found(lab_id)

    llm = request.app.state.llm_client
    issues = await llm.check_typos(
        spec_markdown=material.spec_markdown,
        quiz=material.quiz,
        rubric=material.rubric,
    )
    return CheckTyposResponse(issues_found=len(issues), issues=issues)
