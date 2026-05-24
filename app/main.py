import json
import asyncio
import hmac

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas import OMRProcessPayload, OMRProcessResponse
from app.services.omr_processor import process_omr_file_pages, process_omr_image


omr_processing_slots = asyncio.Semaphore(settings.request_concurrency)

app = FastAPI(
    title="LiEnsina OMR Service",
    version="0.1.0",
    description="Microservico de visao computacional para leitura de cartao resposta.",
    docs_url=None if settings.env == "production" else "/docs",
    redoc_url=None if settings.env == "production" else "/redoc",
    openapi_url=None if settings.env == "production" else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str | int | float]:
    if settings.env == "production":
        return {"status": "ok"}
    return {
        "status": "ok",
        "service": "LiEnsina_OMR_Service",
        "engine": "opencv-threshold-v1",
        "requestConcurrency": settings.request_concurrency,
        "detectedCpuCores": settings.detected_cpu_cores,
        "pageConcurrency": settings.page_concurrency,
        "opencvThreads": settings.opencv_threads,
        "pdfRenderScale": settings.pdf_render_scale,
        "maxPdfPages": settings.max_pdf_pages,
    }


def parse_payload(payload: str) -> OMRProcessPayload:
    if len(payload.encode("utf-8")) > settings.max_payload_bytes:
        raise HTTPException(status_code=413, detail="Payload OMR excede o limite permitido.")
    try:
        data = json.loads(payload)
        return OMRProcessPayload.model_validate(data)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="payload deve ser um JSON valido.") from error
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error


def require_internal_token(token: str | None) -> None:
    if not settings.internal_token:
        if settings.env == "production":
            raise HTTPException(status_code=503, detail="OMR internal auth is not configured.")
        return
    if not token or not hmac.compare_digest(token, settings.internal_token):
        raise HTTPException(status_code=401, detail="Token interno OMR invalido.")


def detect_allowed_file_type(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.lstrip()[:5].lower().startswith(b"<svg") or data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        raise HTTPException(status_code=415, detail="SVG e GIF nao sao aceitos pelo OMR.")
    raise HTTPException(status_code=415, detail="Tipo real do arquivo nao suportado.")


async def run_processing_with_timeout(processor, *args, **kwargs):
    try:
        return await asyncio.wait_for(
            run_in_threadpool(processor, *args, **kwargs),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(status_code=504, detail="OMR_PROCESSING_TIMEOUT") from error


@app.post("/v1/omr/process", response_model=OMRProcessResponse)
async def process_omr(
    payload: str = Form(...),
    image: UploadFile = File(...),
    x_liensina_omr_token: str | None = Header(default=None),
    content_length: int | None = Header(default=None),
) -> OMRProcessResponse:
    require_internal_token(x_liensina_omr_token)
    if content_length is not None and content_length > settings.max_upload_bytes + 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Requisicao excede {settings.max_upload_mb}MB.")
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/pdf"):
        raise HTTPException(status_code=415, detail="Envie uma imagem JPEG, PNG, WEBP ou PDF.")

    parsed_payload = parse_payload(payload)
    try:
        async with omr_processing_slots:
            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Arquivo vazio.")
            if len(image_bytes) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail=f"Arquivo excede {settings.max_upload_mb}MB.")
            detected_content_type = detect_allowed_file_type(image_bytes)

            return await run_processing_with_timeout(
                process_omr_image,
                image_bytes,
                parsed_payload,
                content_type=detected_content_type,
            )
    except ValueError as error:
        status_code = 422 if str(error) == "PDF_PAGE_LIMIT_EXCEEDED" else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.post("/v1/omr/process-batch", response_model=list[OMRProcessResponse])
async def process_omr_batch(
    payload: str = Form(...),
    image: UploadFile = File(...),
    x_liensina_omr_token: str | None = Header(default=None),
    content_length: int | None = Header(default=None),
) -> list[OMRProcessResponse]:
    require_internal_token(x_liensina_omr_token)
    if content_length is not None and content_length > settings.max_upload_bytes + 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Requisicao excede {settings.max_upload_mb}MB.")
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/pdf"):
        raise HTTPException(status_code=415, detail="Envie uma imagem JPEG, PNG, WEBP ou PDF.")

    parsed_payload = parse_payload(payload)
    try:
        async with omr_processing_slots:
            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Arquivo vazio.")
            if len(image_bytes) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail=f"Arquivo excede {settings.max_upload_mb}MB.")
            detected_content_type = detect_allowed_file_type(image_bytes)

            return await run_processing_with_timeout(
                process_omr_file_pages,
                image_bytes,
                parsed_payload,
                content_type=detected_content_type,
            )
    except ValueError as error:
        status_code = 422 if str(error) == "PDF_PAGE_LIMIT_EXCEEDED" else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error
