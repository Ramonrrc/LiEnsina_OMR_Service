import json
import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import OMRProcessPayload, OMRProcessResponse, QRCodeReport, QualityReport
from app.services.omr_processor import validate_qr


client = TestClient(main.app)


def valid_payload(**overrides):
    payload = {
        "examId": "exam-1",
        "versionId": "version-1",
        "answerKey": [
            {"questionNumber": 1, "correctOption": "A"},
            {"questionNumber": 2, "correctOption": "B"},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def fake_response() -> OMRProcessResponse:
    return OMRProcessResponse(
        examId="exam-1",
        versionId="version-1",
        answerCardId=None,
        studentId=None,
        classId=None,
        templateVersion="liensina-omr-v1",
        suggestedScore=0,
        correctCount=0,
        wrongCount=0,
        blankCount=0,
        multipleCount=0,
        totalQuestions=2,
        confidence=0,
        requiresReview=True,
        shouldRetakeImage=True,
        failures=["CARD_ALIGNMENT_FAILED"],
        quality=QualityReport(width=1, height=1, brightness=0, contrast=0, blur=0, warnings=[], confidence=0),
        qrCode=QRCodeReport(found=False, warnings=["QR_NOT_FOUND"]),
        detectedAnswers=[],
        metadata={},
    )


@pytest.fixture(autouse=True)
def configure_internal_token(monkeypatch):
    object.__setattr__(main.settings, "internal_token", "test-token")
    object.__setattr__(main.settings, "request_timeout_seconds", 20)
    object.__setattr__(main.settings, "max_upload_mb", 16)
    yield


def post_process(payload=None, file_bytes=None, token="test-token", content_type="image/png"):
    return client.post(
        "/v1/omr/process",
        data={"payload": payload or valid_payload()},
        files={"image": ("card.png", file_bytes or b"\x89PNG\r\n\x1a\nsafe", content_type)},
        headers={"X-LiEnsina-OMR-Token": token},
    )


def test_internal_token_is_required_before_processing(monkeypatch):
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return fake_response()

    monkeypatch.setattr(main, "process_omr_image", should_not_run)
    response = post_process(token="wrong-token")

    assert response.status_code == 401
    assert called is False


def test_invalid_magic_bytes_are_rejected_before_processing(monkeypatch):
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return fake_response()

    monkeypatch.setattr(main, "process_omr_image", should_not_run)
    response = post_process(file_bytes=b"not-a-real-image", content_type="image/png")

    assert response.status_code == 415
    assert called is False


def test_svg_payload_is_rejected():
    response = post_process(file_bytes=b"<svg><script>alert(1)</script></svg>", content_type="image/svg+xml")

    assert response.status_code == 415


def test_payload_extra_fields_are_rejected(monkeypatch):
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return fake_response()

    monkeypatch.setattr(main, "process_omr_image", should_not_run)
    response = post_process(payload=valid_payload(role="ADMIN"))

    assert response.status_code == 422
    assert called is False


def test_pdf_page_limit_error_returns_422(monkeypatch):
    def raise_page_limit(*_args, **_kwargs):
        raise ValueError("PDF_PAGE_LIMIT_EXCEEDED")

    monkeypatch.setattr(main, "process_omr_file_pages", raise_page_limit)
    response = client.post(
        "/v1/omr/process-batch",
        data={"payload": valid_payload()},
        files={"image": ("cards.pdf", b"%PDF-1.7\nsafe", "application/pdf")},
        headers={"X-LiEnsina-OMR-Token": "test-token"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "PDF_PAGE_LIMIT_EXCEEDED"


def test_processing_timeout_returns_504(monkeypatch):
    object.__setattr__(main.settings, "request_timeout_seconds", 0.01)

    def slow_processor(*_args, **_kwargs):
        time.sleep(0.05)
        return fake_response()

    monkeypatch.setattr(main, "process_omr_image", slow_processor)
    response = post_process()

    assert response.status_code == 504


def test_qr_from_other_exam_is_flagged():
    payload = OMRProcessPayload.model_validate(
        {
            "examId": "exam-1",
            "versionId": "version-1",
            "answerKey": [{"questionNumber": 1, "correctOption": "A"}],
        }
    )
    qr = QRCodeReport(found=True, raw="{}", parsed={"examId": "other-exam", "versionId": "other-version"})

    warnings = validate_qr(payload, qr)

    assert "QR_EXAM_ID_MISMATCH" in warnings
    assert "QR_VERSION_ID_MISMATCH" in warnings


def test_production_requires_strong_internal_token():
    env = {**os.environ, "OMR_ENV": "production"}
    env.pop("OMR_INTERNAL_TOKEN", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.core.config"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "OMR_INTERNAL_TOKEN" in (result.stderr + result.stdout)
