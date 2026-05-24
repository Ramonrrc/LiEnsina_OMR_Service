from __future__ import annotations

import json
import cv2
import numpy as np

from app.schemas import QRCodeReport

try:
    from pyzbar.pyzbar import decode
except Exception:  # pragma: no cover - pyzbar depends on libzbar at runtime.
    decode = None


def build_report_from_raw(raw: str) -> QRCodeReport:
    parsed = None
    warnings: list[str] = []
    try:
        parsed_value = json.loads(raw)
        if isinstance(parsed_value, dict):
            parsed = parsed_value
        else:
            warnings.append("QR_PAYLOAD_NOT_OBJECT")
    except json.JSONDecodeError:
        warnings.append("QR_PAYLOAD_NOT_JSON")

    return QRCodeReport(found=True, raw=raw, parsed=parsed, warnings=warnings)


def qr_preprocessing_variants(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    for source in [gray, clahe]:
        variants.append(cv2.GaussianBlur(source, (3, 3), 0))
        _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        adaptive = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        variants.append(adaptive)

    max_side = max(gray.shape[:2])
    if max_side > 700:
        return variants

    resized: list[np.ndarray] = []
    for variant in variants:
        pad = max(16, int(min(variant.shape[:2]) * 0.1))
        resized.append(variant)
        resized.append(cv2.copyMakeBorder(variant, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255))
        for scale in (1.5, 2.0):
            scaled = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            scaled_pad = max(24, int(min(scaled.shape[:2]) * 0.08))
            resized.append(cv2.copyMakeBorder(scaled, scaled_pad, scaled_pad, scaled_pad, scaled_pad, cv2.BORDER_CONSTANT, value=255))
    return resized


def decode_with_pyzbar(image: np.ndarray) -> QRCodeReport | None:
    if decode is None:
        return None

    for variant in qr_preprocessing_variants(image):
        decoded = decode(variant)
        if decoded:
            raw = decoded[0].data.decode("utf-8", errors="replace")
            return build_report_from_raw(raw)
    return None


def decode_with_opencv(image: np.ndarray) -> QRCodeReport | None:
    detector = cv2.QRCodeDetector()
    detected_points = None

    for variant in qr_preprocessing_variants(image):
        try:
            data, points, _straight = detector.detectAndDecode(variant)
        except cv2.error:
            data, points = "", None
        if points is not None:
            detected_points = points
        if data:
            return build_report_from_raw(data)

        detect_curved = getattr(detector, "detectAndDecodeCurved", None)
        if callable(detect_curved):
            try:
                data, points, _straight = detect_curved(variant)
            except cv2.error:
                data, points = "", None
            if points is not None:
                detected_points = points
            if data:
                return build_report_from_raw(data)

    if detected_points is not None:
        return QRCodeReport(found=True, raw=None, parsed=None, warnings=["QR_DETECTED_BUT_UNREADABLE"])
    return None


def read_qr_code(image: np.ndarray) -> QRCodeReport:
    warnings = ["QR_NOT_FOUND"]
    if decode is not None:
        report = decode_with_pyzbar(image)
        if report:
            return report

    report = decode_with_opencv(image)
    if report:
        return report

    if decode is None:
        warnings.append("QR_DECODER_UNAVAILABLE")
    return QRCodeReport(found=False, warnings=warnings)
