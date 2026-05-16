import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.core.config import settings
from app.schemas import OMRProcessPayload, OMRProcessResponse
from app.services.omr_processor import process_omr_image


app = FastAPI(
    title="LiEnsina OMR Service",
    version="0.1.0",
    description="Microservico de visao computacional para leitura de cartao resposta.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "LiEnsina_OMR_Service", "engine": "opencv-threshold-v1"}


def parse_payload(payload: str) -> OMRProcessPayload:
    try:
        data = json.loads(payload)
        return OMRProcessPayload.model_validate(data)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="payload deve ser um JSON valido.") from error
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error


@app.post("/v1/omr/process", response_model=OMRProcessResponse)
async def process_omr(
    payload: str = Form(...),
    image: UploadFile = File(...),
) -> OMRProcessResponse:
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/pdf"):
        raise HTTPException(status_code=415, detail="Envie uma imagem JPEG, PNG, WEBP ou PDF.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"Arquivo excede {settings.max_upload_mb}MB.")

    parsed_payload = parse_payload(payload)
    try:
        return process_omr_image(image_bytes, parsed_payload, content_type=content_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
