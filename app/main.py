import json
import asyncio
import hmac
from multiprocessing import get_context
from queue import Empty
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas import OMRProcessPayload, OMRProcessResponse
from app.services.omr_processor import process_omr_file_pages, process_omr_image


omr_processing_slots = asyncio.Semaphore(settings.request_concurrency)
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

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


def _serialize_processor_result(value: Any) -> Any:
    if isinstance(value, list):
        return [_serialize_processor_result(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _run_isolated_processor(processor_name: str, queue, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        processor = process_omr_file_pages if processor_name == "batch" else process_omr_image
        queue.put(("ok", _serialize_processor_result(processor(*args, **kwargs))))
    except Exception as error:  # noqa: BLE001 - serialized back to the API process safely.
        queue.put(("error", error.__class__.__name__, str(error)))


async def run_processing_in_isolated_process(processor_name: str, *args, **kwargs):
    context = get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_run_isolated_processor, args=(processor_name, queue, args, kwargs))
    process.start()
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(loop.run_in_executor(None, process.join), timeout=settings.request_timeout_seconds)
    except asyncio.TimeoutError as error:
        process.terminate()
        await loop.run_in_executor(None, process.join)
        raise HTTPException(status_code=504, detail="OMR_PROCESSING_TIMEOUT") from error

    if process.exitcode not in (0, None):
        raise HTTPException(status_code=500, detail="OMR_PROCESSING_FAILED")

    try:
        status, *payload = queue.get_nowait()
    except Empty as error:
        raise HTTPException(status_code=500, detail="OMR_PROCESSING_FAILED") from error

    if status == "error":
        error_type, message = payload
        if error_type == "ValueError":
            raise ValueError(message)
        raise HTTPException(status_code=400, detail=message)

    result = payload[0]
    if processor_name == "batch":
        return [OMRProcessResponse.model_validate(item) for item in result]
    return OMRProcessResponse.model_validate(result)


async def run_processing_with_timeout(processor_name: str, processor, *args, **kwargs):
    if settings.process_isolation:
        return await run_processing_in_isolated_process(processor_name, *args, **kwargs)
    try:
        return await asyncio.wait_for(
            run_in_threadpool(processor, *args, **kwargs),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(status_code=504, detail="OMR_PROCESSING_TIMEOUT") from error


async def read_upload_bytes_limited(image: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Arquivo excede {settings.max_upload_mb}MB.")
        chunks.append(chunk)
    return b"".join(chunks)


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
            image_bytes = await read_upload_bytes_limited(image)
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Arquivo vazio.")
            detected_content_type = detect_allowed_file_type(image_bytes)

            return await run_processing_with_timeout(
                "single",
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
            image_bytes = await read_upload_bytes_limited(image)
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Arquivo vazio.")
            detected_content_type = detect_allowed_file_type(image_bytes)

            return await run_processing_with_timeout(
                "batch",
                process_omr_file_pages,
                image_bytes,
                parsed_payload,
                content_type=detected_content_type,
            )
    except ValueError as error:
        status_code = 422 if str(error) == "PDF_PAGE_LIMIT_EXCEEDED" else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error
