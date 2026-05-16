import cv2
import numpy as np

from app.schemas import QualityReport


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def evaluate_image_quality(image: np.ndarray) -> QualityReport:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings: list[str] = []
    if min(width, height) < 700:
        warnings.append("IMAGE_TOO_SMALL")
    if brightness < 55:
        warnings.append("LOW_LIGHT")
    if brightness > 220:
        warnings.append("OVEREXPOSED")
    if contrast < 25:
        warnings.append("LOW_CONTRAST")
    if blur < 70:
        warnings.append("BLURRY_IMAGE")

    resolution_score = clamp((min(width, height) - 450) / 900)
    brightness_score = clamp(1 - abs(brightness - 135) / 140)
    contrast_score = clamp((contrast - 12) / 58)
    blur_score = clamp((blur - 35) / 180)
    confidence = round(
        0.30 * resolution_score
        + 0.25 * brightness_score
        + 0.20 * contrast_score
        + 0.25 * blur_score,
        4,
    )

    return QualityReport(
        width=width,
        height=height,
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        blur=round(blur, 2),
        warnings=warnings,
        confidence=confidence,
    )
