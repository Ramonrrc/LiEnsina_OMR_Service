from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import gc
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium

from app.core.config import settings
from app.schemas import (
    BubbleOptionScore,
    DetectedAnswer,
    OMRProcessPayload,
    OMRProcessResponse,
    QRCodeReport,
)
from app.services.image_quality import clamp, evaluate_image_quality
from app.services.layout import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    build_grid_layout,
    bubble_positions,
    marker_centers,
)
from app.services.qrcode_reader import read_qr_code

cv2.setUseOptimized(True)
cv2.setNumThreads(settings.opencv_threads)


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("IMAGE_DECODE_FAILED")
    return image


def decode_pdf_first_page(pdf_bytes: bytes) -> np.ndarray:
    try:
        document = pdfium.PdfDocument(pdf_bytes)
    except Exception as error:
        raise ValueError("PDF_DECODE_FAILED") from error

    try:
        page_count = len(document)
        if page_count < 1:
            raise ValueError("PDF_EMPTY")
        if page_count > settings.max_pdf_pages:
            raise ValueError("PDF_PAGE_LIMIT_EXCEEDED")

        page = document[0]
        try:
            bitmap = page.render(scale=settings.pdf_render_scale)
            pil_image = bitmap.to_pil().convert("RGB")
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()

        rgb = np.array(pil_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        close_document = getattr(document, "close", None)
        if callable(close_document):
            close_document()


def decode_pdf_pages(pdf_bytes: bytes) -> list[np.ndarray]:
    try:
        document = pdfium.PdfDocument(pdf_bytes)
    except Exception as error:
        raise ValueError("PDF_DECODE_FAILED") from error

    try:
        page_count = len(document)
        if page_count < 1:
            raise ValueError("PDF_EMPTY")
        if page_count > settings.max_pdf_pages:
            raise ValueError("PDF_PAGE_LIMIT_EXCEEDED")

        pages: list[np.ndarray] = []
        for index in range(page_count):
            page = document[index]
            try:
                bitmap = page.render(scale=settings.pdf_render_scale)
                pil_image = bitmap.to_pil().convert("RGB")
            finally:
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()

            rgb = np.array(pil_image)
            pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return pages
    finally:
        close_document = getattr(document, "close", None)
        if callable(close_document):
            close_document()


def decode_omr_file(file_bytes: bytes, content_type: str | None = None) -> np.ndarray:
    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type == "application/pdf" or file_bytes.startswith(b"%PDF"):
        return decode_pdf_first_page(file_bytes)
    return decode_image(file_bytes)


def decode_omr_pages(file_bytes: bytes, content_type: str | None = None) -> list[np.ndarray]:
    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type == "application/pdf" or file_bytes.startswith(b"%PDF"):
        return decode_pdf_pages(file_bytes)
    return [decode_image(file_bytes)]


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def find_marker_quad(image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray]] = []
    image_area = image.shape[0] * image.shape[1]
    min_dimension = min(image.shape[:2])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.00015:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if w > image.shape[1] * 0.18 or h > image.shape[0] * 0.18:
            continue
        if min(w, h) < min_dimension * 0.018 or max(w, h) > min_dimension * 0.12:
            continue
        ratio = w / h
        if ratio < 0.55 or ratio > 1.75:
            continue
        extent = area / float(w * h)
        if extent < 0.60:
            continue
        candidates.append((area, np.array([x + w / 2, y + h / 2], dtype="float32")))

    if len(candidates) >= 4:
        largest = sorted(candidates, key=lambda item: item[0], reverse=True)[:12]
        centers = np.array([item[1] for item in largest], dtype="float32")
        top_left = centers[np.argmin(centers[:, 0] + centers[:, 1])]
        top_right = centers[np.argmax(centers[:, 0] - centers[:, 1])]
        bottom_right = centers[np.argmax(centers[:, 0] + centers[:, 1])]
        bottom_left = centers[np.argmax(centers[:, 1] - centers[:, 0])]
        quad = order_points(np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"))
        side_lengths = [
            float(np.linalg.norm(quad[1] - quad[0])),
            float(np.linalg.norm(quad[2] - quad[1])),
            float(np.linalg.norm(quad[2] - quad[3])),
            float(np.linalg.norm(quad[3] - quad[0])),
        ]
        has_distinct_corners = len(np.unique(np.round(quad, 1), axis=0)) == 4
        has_readable_sides = min(side_lengths) > min_dimension * 0.18
        if has_distinct_corners and has_readable_sides and cv2.contourArea(quad) > image_area * 0.12:
            return quad, None

    document_quad = find_document_quad(image)
    if document_quad is not None:
        return document_quad, "ALIGNMENT_MARKERS_NOT_FOUND_USING_DOCUMENT_CONTOUR"
    return None, "CARD_NOT_FOUND"


def find_document_quad(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image_area * 0.25:
            return order_points(approx.reshape(4, 2).astype("float32"))
    return None


def warp_card(image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    source_quad, warning = find_marker_quad(image)
    if source_quad is None:
        return None, warning

    if warning == "ALIGNMENT_MARKERS_NOT_FOUND_USING_DOCUMENT_CONTOUR":
        destination = np.array(
            [
                (0, 0),
                (CANONICAL_WIDTH - 1, 0),
                (CANONICAL_WIDTH - 1, CANONICAL_HEIGHT - 1),
                (0, CANONICAL_HEIGHT - 1),
            ],
            dtype="float32",
        )
    else:
        destination = np.array(marker_centers(), dtype="float32")
    matrix = cv2.getPerspectiveTransform(source_quad, destination)
    warped = cv2.warpPerspective(image, matrix, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
    return warped, warning


def dark_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold


def count_dark_components(image: np.ndarray) -> int:
    threshold = dark_threshold(image)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(8, image.shape[0] * image.shape[1] * 0.00004)
    return sum(1 for contour in contours if cv2.contourArea(contour) >= min_area)


def dark_ratio(image: np.ndarray) -> float:
    threshold = dark_threshold(image)
    return cv2.countNonZero(threshold) / max(threshold.size, 1)


def qr_report_rank(report: QRCodeReport) -> int:
    if report.parsed:
        return 3
    if report.raw:
        return 2
    if report.found:
        return 1
    return 0


def best_qr_report(*reports: QRCodeReport) -> QRCodeReport:
    return max(reports, key=qr_report_rank)


def skipped_qr_report() -> QRCodeReport:
    return QRCodeReport(found=False, raw=None, parsed=None, warnings=["QR_SKIPPED"])


def read_card_qr_code(card: np.ndarray) -> QRCodeReport:
    reports: list[QRCodeReport] = []
    primary_regions = [
        (45, 330, 705, 1035),
        (50, 310, 720, 990),
        (60, 300, 735, 975),
        (45, 315, 710, 980),
    ]
    fallback_regions = [
        (50, 320, 725, 985),
        (45, 330, 705, 1035),
        (40, 300, 740, 1010),
        (30, 350, 680, 1040),
        (72, 262, 780, 970),
        (55, 285, 755, 995),
        (20, 360, 650, 1060),
    ]

    for y1, y2, x1, x2 in primary_regions:
        crop = card[y1:y2, x1:x2]
        if crop.size:
            report = read_qr_code(crop)
            if report.parsed:
                return report
            reports.append(report)

    full_report = read_qr_code(card)
    if full_report.parsed:
        return full_report
    reports.append(full_report)

    for y1, y2, x1, x2 in fallback_regions:
        crop = card[y1:y2, x1:x2]
        if crop.size:
            report = read_qr_code(crop)
            if report.parsed:
                return report
            reports.append(report)

    return best_qr_report(*reports)


def qr_region_score(card: np.ndarray, qr_report: QRCodeReport | None = None) -> float:
    if qr_report is not None and qr_report.parsed:
        return 8.0
    if qr_report is not None and "QR_SKIPPED" in qr_report.warnings:
        crop = card[45:330, 705:1035]
        if crop.size == 0:
            return 0.0
        components = count_dark_components(crop)
        density = dark_ratio(crop)
        return min(components / 45.0, 3.0) + min(max(density - 0.06, 0.0) * 12.0, 2.0)

    crop = card[45:330, 705:1035]
    if crop.size == 0:
        return 0.0

    report = read_qr_code(crop)
    components = count_dark_components(crop)
    density = dark_ratio(crop)
    if report.parsed:
        return 8.0
    if report.found and components >= 25 and density >= 0.08:
        return 6.0

    return min(components / 45.0, 3.0) + min(max(density - 0.06, 0.0) * 12.0, 2.0)


def header_region_score(card: np.ndarray) -> float:
    crop = card[60:260, 150:700]
    if crop.size == 0:
        return 0.0

    components = count_dark_components(crop)
    density = dark_ratio(crop)
    return min(components / 14.0, 3.0) + min(max(density - 0.015, 0.0) * 20.0, 2.0)


def card_orientation_score(card: np.ndarray, qr_report: QRCodeReport | None = None) -> float:
    return qr_region_score(card, qr_report) * 1.7 + header_region_score(card)


def orient_warped_card(card: np.ndarray) -> tuple[np.ndarray, str | None]:
    rotated = cv2.rotate(card, cv2.ROTATE_180)
    score_normal = card_orientation_score(card)
    score_rotated = card_orientation_score(rotated)

    if score_rotated > score_normal + 1.25:
        return rotated, "ROTATE_180"
    return card, None


def read_bubble_fill(image: np.ndarray, x: int, y: int, radius: int) -> float:
    pad = int(radius * 1.65)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image.shape[1], x + pad), min(image.shape[0], y + pad)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros_like(threshold)
    center = (int(np.clip(x - x1, 0, roi.shape[1] - 1)), int(np.clip(y - y1, 0, roi.shape[0] - 1)))
    cv2.circle(mask, center, radius, 255, -1)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(threshold, threshold, mask=mask))
    mask_pixels = cv2.countNonZero(mask)
    if mask_pixels == 0:
        return 0.0

    inner_mask = np.zeros_like(threshold)
    cv2.circle(inner_mask, center, max(4, int(radius * 0.68)), 255, -1)
    inner_pixels = cv2.countNonZero(inner_mask)
    if inner_pixels == 0:
        return round(float(dark_pixels / mask_pixels), 4)

    outer_mask = np.zeros_like(threshold)
    cv2.circle(outer_mask, center, min(pad, int(radius * 1.45)), 255, -1)
    ring_mask = cv2.subtract(outer_mask, mask)
    ring_values = gray[ring_mask > 0]
    inner_values = gray[inner_mask > 0]
    background = float(np.median(ring_values)) if ring_values.size else float(np.median(gray))
    local_std = float(np.std(ring_values)) if ring_values.size else float(np.std(gray))
    darkness_margin = max(12.0, local_std * 0.55)
    relative_dark = float(np.mean(inner_values < background - darkness_margin)) if inner_values.size else 0.0

    inner_dark_pixels = cv2.countNonZero(cv2.bitwise_and(threshold, threshold, mask=inner_mask))
    inner_dark = float(inner_dark_pixels / inner_pixels)
    legacy_dark = float(dark_pixels / mask_pixels)

    saturation_fill = 0.0
    if roi.ndim == 3:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        ring_saturation = saturation[ring_mask > 0]
        saturation_background = float(np.median(ring_saturation)) if ring_saturation.size else 0.0
        saturated_ink = (
            (saturation > max(38.0, saturation_background + 22.0))
            & (value < 248)
            & (inner_mask > 0)
        )
        saturation_fill = float(np.count_nonzero(saturated_ink) / inner_pixels)

    fill_ratio = max(
        legacy_dark * 0.88,
        inner_dark * 1.22,
        relative_dark * 1.08,
        saturation_fill * 1.18,
    )
    return round(float(clamp(fill_ratio, 0.0, 1.0)), 4)


def classify_question_answer(
    question_number: int,
    key_item: Any,
    option_scores: list[BubbleOptionScore],
) -> DetectedAnswer:
    sorted_scores = sorted(option_scores, key=lambda score: score.fillRatio, reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else BubbleOptionScore(option="A", fillRatio=0)
    marked = [
        score.option
        for score in option_scores
        if score.fillRatio >= max(0.24, top.fillRatio * 0.72)
    ]

    status = "ok"
    detected_option: str | None = top.option
    if top.fillRatio < 0.20:
        status = "blank"
        detected_option = None
        marked = []
    elif len(marked) > 1:
        status = "multiple"
        detected_option = None
    elif top.fillRatio - second.fillRatio < 0.075:
        status = "low_confidence"

    confidence = clamp((top.fillRatio - second.fillRatio) * 3.2 + (top.fillRatio - 0.18) * 1.45, 0.05, 0.99)
    return DetectedAnswer(
        questionNumber=question_number,
        questionId=key_item.questionId,
        detectedOption=detected_option,
        correctOption=key_item.correctOption,
        isCorrect=detected_option == key_item.correctOption and status in {"ok", "low_confidence"},
        status=status,
        confidence=round(confidence, 4),
        markedOptions=marked,
        optionScores=option_scores,
    )


def classify_answers(warped: np.ndarray, payload: OMRProcessPayload) -> list[DetectedAnswer]:
    image = cv2.GaussianBlur(warped, (3, 3), 0)
    positions_by_question: dict[int, list[tuple[str, int, int, int]]] = defaultdict(list)
    for position in bubble_positions(len(payload.answerKey)):
        positions_by_question[position.question_number].append(
            (position.option, position.x, position.y, position.radius)
        )

    detected_answers: list[DetectedAnswer] = []
    key_by_number = {item.questionNumber: item for item in payload.answerKey}
    for question_number in range(1, len(payload.answerKey) + 1):
        option_scores: list[BubbleOptionScore] = []
        for option, x, y, radius in positions_by_question[question_number]:
            option_scores.append(
                BubbleOptionScore(option=option, fillRatio=read_bubble_fill(image, x, y, radius))
            )

        detected_answers.append(classify_question_answer(question_number, key_by_number[question_number], option_scores))

    return sorted(detected_answers, key=lambda answer: answer.questionNumber)


def cluster_axis(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - float(np.mean(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def find_bubble_candidates(image: np.ndarray) -> list[dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_dimension = min(image.shape[:2])
    min_size = max(18, min_dimension * 0.015)
    max_size = min_dimension * 0.08
    circles: list[dict[str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue
        ratio = w / h
        if ratio < 0.72 or ratio > 1.38:
            continue
        extent = area / float(w * h)
        if extent < 0.42 or extent > 0.90:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        if circularity < 0.60:
            continue
        circles.append({
            "x": x + w / 2,
            "y": y + h / 2,
            "radius": max(8.0, min(w, h) * 0.44),
        })

    return circles


def find_bubble_grid_candidates(image: np.ndarray) -> list[dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_dimension = min(image.shape[:2])
    min_size = max(8, min_dimension * 0.006)
    max_size = min_dimension * 0.08
    min_area = max(140.0, min_dimension * min_dimension * 0.00016)
    max_area = min_dimension * min_dimension * 0.0022
    candidates: list[dict[str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue
        ratio = w / h
        if ratio < 0.25 or ratio > 2.20:
            continue
        if area < min_area or area > max_area:
            continue
        extent = area / float(w * h)
        if extent < 0.28 or extent > 0.94:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        if circularity < 0.35:
            continue
        candidates.append({
            "x": x + w / 2,
            "y": y + h / 2,
            "radius": max(8.0, min(w, h) * 0.48),
        })

    return candidates


def select_regular_window(
    items: list[tuple[float, Any, int]],
    expected_size: int,
    *,
    min_spacing: float,
    max_spacing: float,
    spacing_penalty: float,
) -> list[tuple[float, Any, int]]:
    if len(items) <= expected_size:
        return items[:expected_size]

    scored_windows: list[tuple[float, list[tuple[float, Any, int]]]] = []
    for start in range(0, len(items) - expected_size + 1):
        window = items[start:start + expected_size]
        centers = [item[0] for item in window]
        spacings = np.diff(centers)
        mean_spacing = float(np.mean(spacings)) if len(spacings) else 0.0
        if mean_spacing < min_spacing or mean_spacing > max_spacing:
            continue
        spacing_std = float(np.std(spacings)) if len(spacings) else 0.0
        count_score = sum(item[2] for item in window)
        scored_windows.append((count_score - spacing_std * spacing_penalty - start * 3.0, window))

    if not scored_windows:
        return items[:expected_size]
    return max(scored_windows, key=lambda item: item[0])[1]


def select_regular_center_groups(
    centers_with_count: list[tuple[float, int]],
    group_size: int,
    group_count: int,
) -> list[float]:
    centers_with_count = sorted(centers_with_count, key=lambda item: item[0])
    if len(centers_with_count) <= group_size * group_count:
        return [item[0] for item in centers_with_count[:group_size * group_count]]

    candidates: list[tuple[float, set[int], list[float]]] = []
    for start in range(0, len(centers_with_count) - group_size + 1):
        window = centers_with_count[start:start + group_size]
        centers = [item[0] for item in window]
        spacings = np.diff(centers)
        mean_spacing = float(np.mean(spacings)) if len(spacings) else 0.0
        if mean_spacing < 18.0 or mean_spacing > 120.0:
            continue
        spacing_std = float(np.std(spacings)) if len(spacings) else 0.0
        counts = [item[1] for item in window]
        count_score = sum(counts)
        score = count_score - spacing_std * 0.7 - float(np.std(counts)) * 4.0 + centers[0] * 0.08
        candidates.append((score, set(range(start, start + group_size)), centers))

    selected: list[tuple[float, list[float]]] = []
    used_indexes: set[int] = set()
    for score, indexes, centers in sorted(candidates, key=lambda item: item[0], reverse=True):
        if indexes & used_indexes:
            continue
        selected.append((score, centers))
        used_indexes.update(indexes)
        if len(selected) >= group_count:
            break

    if len(selected) < group_count:
        fallback = sorted(centers_with_count, key=lambda item: item[1], reverse=True)[:group_size * group_count]
        return sorted([item[0] for item in fallback])

    return sorted(center for _, centers in selected for center in centers)


def classify_answers_from_centers(
    image: np.ndarray,
    payload: OMRProcessPayload,
    row_centers: list[float],
    x_centers: list[float],
    radius: float,
) -> list[DetectedAnswer]:
    key_by_number = {item.questionNumber: item for item in payload.answerKey}
    detected_answers: list[DetectedAnswer] = []
    for row_index, row_center_y in enumerate(row_centers):
        question_number = row_index + 1
        if question_number > len(payload.answerKey):
            break

        option_scores: list[BubbleOptionScore] = []
        for option, center_x in zip(payload.options, x_centers):
            option_scores.append(
                BubbleOptionScore(
                    option=option,
                    fillRatio=read_bubble_fill(
                        image,
                        int(round(center_x)),
                        int(round(row_center_y)),
                        int(round(radius)),
                    ),
                )
            )

        detected_answers.append(classify_question_answer(question_number, key_by_number[question_number], option_scores))

    return sorted(detected_answers, key=lambda answer: answer.questionNumber)


def find_horizontal_separator_y(image: np.ndarray) -> float | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = threshold.shape
    x1, x2 = int(width * 0.08), int(width * 0.92)
    center_band = threshold[:, x1:x2]
    kernel_width = max(90, int(width * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(center_band, cv2.MORPH_OPEN, kernel)

    search_top = int(height * 0.16)
    search_bottom = int(height * 0.44)
    row_strength = np.count_nonzero(horizontal[search_top:search_bottom], axis=1)
    if row_strength.size == 0:
        return None

    best_offset = int(np.argmax(row_strength))
    best_strength = int(row_strength[best_offset])
    if best_strength < (x2 - x1) * 0.28:
        return None
    return float(search_top + best_offset)


def infer_legacy_option_centers(image: np.ndarray, separator_y: float, option_count: int) -> list[float]:
    height, width = image.shape[:2]
    label_top = int(max(0, separator_y + height * 0.045))
    label_bottom = int(min(height, separator_y + height * 0.095))
    label_band = image[label_top:label_bottom, int(width * 0.18):int(width * 0.82)]
    if label_band.size == 0:
        scale_x = width / CANONICAL_WIDTH
        return [value * scale_x for value in (416, 493, 570, 647, 724)[:option_count]]

    gray = cv2.cvtColor(label_band, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_offset = int(width * 0.18)
    centers: list[tuple[float, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, _y, w, h = cv2.boundingRect(contour)
        if area < 20 or area > 600:
            continue
        if w < 5 or w > 42 or h < 8 or h > 45:
            continue
        center_x = float(x_offset + x + w / 2)
        if width * 0.25 <= center_x <= width * 0.75:
            centers.append((center_x, 1))

    if len(centers) >= option_count:
        clustered = [
            (float(np.mean(cluster)), len(cluster))
            for cluster in cluster_axis([center for center, _ in centers], max(14.0, width * 0.012))
        ]
        clustered = [
            item
            for item in clustered
            if width * 0.25 <= item[0] <= width * 0.75
        ]
        if len(clustered) >= option_count:
            selected = select_regular_window(
                [(center, None, count) for center, count in clustered],
                option_count,
                min_spacing=width * 0.035,
                max_spacing=width * 0.095,
                spacing_penalty=0.7,
            )
            if len(selected) == option_count:
                return [center for center, _unused, _count in selected]

    scale_x = width / CANONICAL_WIDTH
    return [value * scale_x for value in (416, 493, 570, 647, 724)[:option_count]]


def detect_legacy_centered_answers(image: np.ndarray, payload: OMRProcessPayload) -> list[DetectedAnswer]:
    total_questions = len(payload.answerKey)
    option_count = len(payload.options)
    if build_grid_layout(total_questions).columns != 1 or option_count != 5:
        return []

    height, width = image.shape[:2]
    if abs(width - CANONICAL_WIDTH) > CANONICAL_WIDTH * 0.18 or abs(height - CANONICAL_HEIGHT) > CANONICAL_HEIGHT * 0.18:
        return []

    separator_y = find_horizontal_separator_y(image)
    if separator_y is None:
        return []

    scale_y = height / CANONICAL_HEIGHT
    if separator_y < 350 * scale_y:
        return []

    row_top = separator_y + 170 * scale_y
    row_spacing = (height - row_top - 140 * scale_y) / max(total_questions - 1, 1)
    row_spacing = float(clamp(row_spacing, 32 * scale_y, 92 * scale_y))
    row_centers = [row_top + row_index * row_spacing for row_index in range(total_questions)]
    if row_centers[-1] > height - 75 * scale_y:
        return []

    x_centers = infer_legacy_option_centers(image, separator_y, option_count)
    radius = 20 * min(width / CANONICAL_WIDTH, scale_y)
    return classify_answers_from_centers(image, payload, row_centers, x_centers, radius)


def detect_bubble_grid_answers(image: np.ndarray, payload: OMRProcessPayload) -> list[DetectedAnswer]:
    circles = find_bubble_grid_candidates(image)
    total_questions = len(payload.answerKey)
    option_count = len(payload.options)
    layout = build_grid_layout(total_questions)

    if len(circles) < total_questions * max(option_count - 1, 3):
        return []

    median_radius = float(np.median([item["radius"] for item in circles]))
    y_tolerance = max(10.0, median_radius * 1.15)
    x_tolerance = max(18.0, median_radius * 1.6)
    y_clusters = cluster_axis([item["y"] for item in circles], y_tolerance)

    row_candidates: list[tuple[float, list[dict[str, float]]]] = []
    min_row_items = max(3, int(layout.columns * option_count * 0.55))
    for cluster in y_clusters:
        center_y = float(np.mean(cluster))
        group = [item for item in circles if abs(item["y"] - center_y) <= y_tolerance]
        if len(group) >= min_row_items:
            row_candidates.append((center_y, sorted(group, key=lambda item: item["x"])))

    row_candidates.sort(key=lambda item: item[0])
    if len(row_candidates) < layout.rows_per_column:
        return []

    if len(row_candidates) > layout.rows_per_column:
        row_items = [(center_y, group, len(group)) for center_y, group in row_candidates]
        row_candidates = [
            (center_y, group)
            for center_y, group, _ in select_regular_window(
                row_items,
                layout.rows_per_column,
                min_spacing=max(12.0, median_radius * 1.6),
                max_spacing=130.0,
                spacing_penalty=0.9,
            )
        ]
    else:
        row_candidates = row_candidates[:layout.rows_per_column]

    row_centers = [item[0] for item in row_candidates]
    row_spacings = np.diff(row_centers)
    if len(row_spacings):
        mean_row_spacing = float(np.mean(row_spacings))
        if (
            float(np.std(row_spacings)) > mean_row_spacing * 0.35
            or float(np.max(row_spacings)) > mean_row_spacing * 1.65
        ):
            return []

    x_clusters = cluster_axis(
        [circle["x"] for _, group in row_candidates for circle in group],
        x_tolerance,
    )
    expected_x_columns = layout.columns * option_count
    min_cluster_items = max(1, int(len(row_candidates) * 0.20))
    x_centers_with_count = [
        (float(np.mean(cluster)), len(cluster))
        for cluster in x_clusters
        if len(cluster) >= min_cluster_items
    ]
    if len(x_centers_with_count) < expected_x_columns:
        x_centers_with_count = sorted(
            [(float(np.mean(cluster)), len(cluster)) for cluster in x_clusters],
            key=lambda item: item[1],
            reverse=True,
        )[:expected_x_columns]

    if len(x_centers_with_count) < expected_x_columns:
        return []

    x_centers = select_regular_center_groups(
        x_centers_with_count,
        option_count,
        layout.columns,
    )
    key_by_number = {item.questionNumber: item for item in payload.answerKey}
    detected_answers: list[DetectedAnswer] = []
    for row_index, (row_center_y, row_circles) in enumerate(row_candidates):
        for column_index in range(layout.columns):
            question_number = column_index * layout.rows_per_column + row_index + 1
            if question_number > total_questions:
                continue

            block_start = column_index * option_count
            block_centers = x_centers[block_start:block_start + option_count]
            option_scores: list[BubbleOptionScore] = []
            for option, center_x in zip(payload.options, block_centers):
                option_scores.append(
                    BubbleOptionScore(
                        option=option,
                        fillRatio=read_bubble_fill(
                            image,
                            int(round(center_x)),
                            int(round(row_center_y)),
                            int(round(max(10.0, median_radius * 1.05))),
                        ),
                    )
                )

            detected_answers.append(
                classify_question_answer(question_number, key_by_number[question_number], option_scores)
            )

    return sorted(detected_answers, key=lambda answer: answer.questionNumber)


def validate_qr(payload: OMRProcessPayload, qr: QRCodeReport) -> list[str]:
    warnings: list[str] = []
    if "QR_SKIPPED" in qr.warnings:
        return warnings
    if not qr.found:
        return ["QR_NOT_FOUND"]
    parsed = qr.parsed or {}
    if parsed.get("examId") and parsed.get("examId") != payload.examId:
        warnings.append("QR_EXAM_ID_MISMATCH")
    if parsed.get("versionId") and parsed.get("versionId") != payload.versionId:
        warnings.append("QR_VERSION_ID_MISMATCH")
    return warnings


def mean_answer_confidence(answers: list[DetectedAnswer]) -> float:
    return float(np.mean([answer.confidence for answer in answers])) if answers else 0.0


def answer_status_counts(answers: list[DetectedAnswer]) -> dict[str, int]:
    return {
        "blank": sum(1 for answer in answers if answer.status == "blank"),
        "multiple": sum(1 for answer in answers if answer.status == "multiple"),
        "low_confidence": sum(1 for answer in answers if answer.status == "low_confidence"),
        "unreadable": sum(1 for answer in answers if answer.status == "unreadable"),
    }


def answer_quality_score(
    answers: list[DetectedAnswer],
    total_questions: int,
    *,
    is_grid: bool = False,
) -> float:
    if len(answers) != total_questions:
        return -1000.0

    counts = answer_status_counts(answers)
    penalty = (
        counts["blank"] * 1.15
        + counts["multiple"] * 2.25
        + counts["low_confidence"] * 0.55
        + counts["unreadable"] * 2.5
    )
    grid_bonus = 0.45 if is_grid else 0.0
    return mean_answer_confidence(answers) * 6.0 + grid_bonus - penalty


def bubble_grid_is_incomplete(bubble_candidate_count: int, payload: OMRProcessPayload) -> bool:
    expected_bubble_count = len(payload.answerKey) * len(payload.options)
    return (
        bubble_candidate_count >= len(payload.answerKey) * max(len(payload.options) - 1, 3)
        and bubble_candidate_count < expected_bubble_count
    )


def analyze_warped_card(
    warped: np.ndarray,
    payload: OMRProcessPayload,
) -> dict[str, Any]:
    total_questions = len(payload.answerKey)
    expected_bubble_count = total_questions * len(payload.options)
    candidates = [
        (None, warped),
        ("ROTATE_180", cv2.rotate(warped, cv2.ROTATE_180)),
    ]
    analyses: list[dict[str, Any]] = []

    for orientation_correction, card in candidates:
        fixed_answers = classify_answers(card, payload)
        grid_answers = detect_bubble_grid_answers(card, payload)
        legacy_answers = detect_legacy_centered_answers(card, payload)
        bubble_candidate_count = len(find_bubble_candidates(card))
        grid_candidate_count = len(find_bubble_grid_candidates(card))
        grid_is_complete = len(grid_answers) == total_questions
        legacy_is_complete = len(legacy_answers) == total_questions
        incomplete_bubble_grid = (
            (not grid_is_complete and not legacy_is_complete and bubble_grid_is_incomplete(bubble_candidate_count, payload))
            or (grid_candidate_count >= expected_bubble_count and not grid_is_complete and not legacy_is_complete)
        )

        fixed_score = answer_quality_score(fixed_answers, total_questions, is_grid=False)
        grid_score = answer_quality_score(grid_answers, total_questions, is_grid=True)
        legacy_score = answer_quality_score(legacy_answers, total_questions, is_grid=True)
        answer_candidates = [
            (fixed_score, fixed_answers, "marker-template"),
            (grid_score, grid_answers, "marker-bubble-grid"),
            (legacy_score, legacy_answers, "marker-legacy-centered-grid"),
        ]
        answer_score, detected_answers, alignment_mode = max(answer_candidates, key=lambda item: item[0])

        completeness_bonus = min(bubble_candidate_count / max(expected_bubble_count, 1), 1.0) * 0.8
        if incomplete_bubble_grid:
            completeness_bonus -= 1.2

        qr = skipped_qr_report() if payload.skipQr else read_card_qr_code(card)
        qr_bonus = 0.8 if qr.parsed else 0.35 if qr.found else 0.0
        orientation_score = card_orientation_score(card, qr)
        total_score = answer_score + orientation_score * 0.55 + completeness_bonus + qr_bonus
        analyses.append(
            {
                "card": card,
                "orientationCorrection": orientation_correction,
                "detectedAnswers": detected_answers,
                "alignmentMode": alignment_mode,
                "bubbleCandidateCount": bubble_candidate_count,
                "gridCandidateCount": grid_candidate_count,
                "expectedBubbleCount": expected_bubble_count,
                "incompleteBubbleGrid": incomplete_bubble_grid,
                "answerScore": answer_score,
                "orientationScore": orientation_score,
                "totalScore": total_score,
                "qr": qr,
            }
        )

    return max(analyses, key=lambda item: item["totalScore"])


def get_quality_failures(warnings: list[str], content_type: str | None = None) -> list[str]:
    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type == "application/pdf":
        return [warning for warning in warnings if warning != "OVEREXPOSED"]
    return warnings


def process_omr_image(image_bytes: bytes, payload: OMRProcessPayload, content_type: str | None = None) -> OMRProcessResponse:
    image = decode_omr_file(image_bytes, content_type)
    return process_omr_array(image, payload, content_type)


def process_omr_array(image: np.ndarray, payload: OMRProcessPayload, content_type: str | None = None) -> OMRProcessResponse:
    quality = evaluate_image_quality(image)
    warped, alignment_warning = warp_card(image)

    failures: list[str] = []
    if warped is None:
        qr_original = skipped_qr_report() if payload.skipQr else read_qr_code(image)
        if alignment_warning:
            failures.append(alignment_warning)
        detected_answers = detect_bubble_grid_answers(image, payload)
        if detected_answers:
            failures = [warning for warning in validate_qr(payload, qr_original) if warning != "QR_NOT_FOUND"]
            correct_count = sum(1 for answer in detected_answers if answer.isCorrect)
            blank_count = sum(1 for answer in detected_answers if answer.status == "blank")
            multiple_count = sum(1 for answer in detected_answers if answer.status == "multiple")
            low_confidence_count = sum(1 for answer in detected_answers if answer.status == "low_confidence")
            wrong_count = len(detected_answers) - correct_count - blank_count - multiple_count
            row_confidence = float(np.mean([answer.confidence for answer in detected_answers])) if detected_answers else 0.0
            confidence = round(0.35 * quality.confidence + 0.65 * row_confidence, 4)
            if quality.warnings:
                failures.extend(get_quality_failures(quality.warnings, content_type))
            if multiple_count:
                failures.append("MULTIPLE_MARKS_DETECTED")
            if blank_count:
                failures.append("BLANK_ANSWERS_DETECTED")
            if low_confidence_count:
                failures.append("LOW_CONFIDENCE_ANSWERS")

            should_retake = quality.confidence < 0.45 or confidence < 0.42
            return OMRProcessResponse(
                examId=payload.examId,
                versionId=payload.versionId,
                answerCardId=payload.answerCardId,
                studentId=payload.studentId,
                classId=payload.classId,
                templateVersion=payload.templateVersion,
                suggestedScore=round((correct_count / max(len(payload.answerKey), 1)) * 10, 2),
                correctCount=correct_count,
                wrongCount=max(0, wrong_count),
                blankCount=blank_count,
                multipleCount=multiple_count,
                totalQuestions=len(payload.answerKey),
                confidence=confidence,
                requiresReview=True,
                shouldRetakeImage=should_retake,
                failures=sorted(set(failures)),
                quality=quality,
                qrCode=qr_original,
                detectedAnswers=detected_answers,
                metadata={
                    "engine": "opencv-threshold-v1",
                    "alignmentMode": "bubble-grid-fallback",
                    "alignmentWarning": alignment_warning,
                },
            )

        raw_bubble_candidate_count = len(find_bubble_candidates(image))
        raw_grid_candidate_count = len(find_bubble_grid_candidates(image))
        expected_bubble_count = len(payload.answerKey) * len(payload.options)
        if raw_grid_candidate_count >= expected_bubble_count or raw_bubble_candidate_count >= expected_bubble_count * 0.55:
            if quality.warnings:
                failures.extend(get_quality_failures(quality.warnings, content_type))
            failures = [
                warning
                for warning in failures
                if warning not in {"CARD_NOT_FOUND", "CARD_ALIGNMENT_FAILED"}
            ]
            failures.append("BUBBLE_GRID_INCOMPLETE")
            return OMRProcessResponse(
                examId=payload.examId,
                versionId=payload.versionId,
                answerCardId=payload.answerCardId,
                studentId=payload.studentId,
                classId=payload.classId,
                templateVersion=payload.templateVersion,
                suggestedScore=0,
                correctCount=0,
                wrongCount=0,
                blankCount=0,
                multipleCount=0,
                totalQuestions=len(payload.answerKey),
                confidence=round(quality.confidence * 0.25, 4),
                requiresReview=True,
                shouldRetakeImage=True,
                failures=sorted(set(failures)),
                quality=quality,
                qrCode=qr_original,
                detectedAnswers=[],
                metadata={
                    "engine": "opencv-threshold-v1",
                    "alignmentMode": "bubble-grid-incomplete",
                    "alignmentWarning": alignment_warning,
                    "bubbleCandidateCount": raw_bubble_candidate_count,
                    "gridCandidateCount": raw_grid_candidate_count,
                    "expectedBubbleCount": expected_bubble_count,
                },
            )

        return OMRProcessResponse(
            examId=payload.examId,
            versionId=payload.versionId,
            answerCardId=payload.answerCardId,
            studentId=payload.studentId,
            classId=payload.classId,
            templateVersion=payload.templateVersion,
            suggestedScore=0,
            correctCount=0,
            wrongCount=0,
            blankCount=0,
            multipleCount=0,
            totalQuestions=len(payload.answerKey),
            confidence=0,
            requiresReview=True,
            shouldRetakeImage=True,
            failures=[*failures, "CARD_ALIGNMENT_FAILED"],
            quality=quality,
            qrCode=qr_original,
            detectedAnswers=[],
            metadata={"engine": "opencv-threshold-v1"},
        )

    analysis = analyze_warped_card(warped, payload)
    warped = analysis["card"]
    orientation_correction = analysis["orientationCorrection"]
    qr_warped = analysis["qr"]
    qr_original = qr_warped if (payload.skipQr or qr_warped.parsed) else read_qr_code(image)
    qr = best_qr_report(qr_warped, qr_original)
    failures.extend(validate_qr(payload, qr))
    detected_answers = analysis["detectedAnswers"]
    alignment_mode = analysis["alignmentMode"]
    bubble_candidate_count = analysis["bubbleCandidateCount"]
    expected_bubble_count = analysis["expectedBubbleCount"]
    incomplete_bubble_grid = analysis["incompleteBubbleGrid"]

    correct_count = sum(1 for answer in detected_answers if answer.isCorrect)
    blank_count = sum(1 for answer in detected_answers if answer.status == "blank")
    multiple_count = sum(1 for answer in detected_answers if answer.status == "multiple")
    low_confidence_count = sum(1 for answer in detected_answers if answer.status == "low_confidence")
    wrong_count = len(detected_answers) - correct_count - blank_count - multiple_count
    row_confidence = mean_answer_confidence(detected_answers)
    confidence = round(0.35 * quality.confidence + 0.65 * row_confidence, 4)

    if quality.warnings:
        failures.extend(get_quality_failures(quality.warnings, content_type))
    if multiple_count:
        failures.append("MULTIPLE_MARKS_DETECTED")
    if blank_count:
        failures.append("BLANK_ANSWERS_DETECTED")
    if low_confidence_count:
        failures.append("LOW_CONFIDENCE_ANSWERS")
    if incomplete_bubble_grid and len(detected_answers) != len(payload.answerKey):
        failures.append("BUBBLE_GRID_INCOMPLETE")

    should_retake = (
        "CARD_NOT_FOUND" in failures
        or "CARD_ALIGNMENT_FAILED" in failures
        or quality.confidence < 0.45
        or confidence < 0.42
    )
    requires_review = True

    return OMRProcessResponse(
        examId=payload.examId,
        versionId=payload.versionId,
        answerCardId=payload.answerCardId,
        studentId=payload.studentId,
        classId=payload.classId,
        templateVersion=payload.templateVersion,
        suggestedScore=round((correct_count / max(len(payload.answerKey), 1)) * 10, 2),
        correctCount=correct_count,
        wrongCount=max(0, wrong_count),
        blankCount=blank_count,
        multipleCount=multiple_count,
        totalQuestions=len(payload.answerKey),
        confidence=confidence,
        requiresReview=requires_review,
        shouldRetakeImage=should_retake,
        failures=sorted(set(failures)),
        quality=quality,
        qrCode=qr,
        detectedAnswers=detected_answers,
        metadata={
            "engine": "opencv-threshold-v1",
            "alignmentMode": alignment_mode,
            "alignmentWarning": alignment_warning,
            "orientationCorrection": orientation_correction,
            "bubbleCandidateCount": bubble_candidate_count,
            "gridCandidateCount": analysis.get("gridCandidateCount"),
            "expectedBubbleCount": expected_bubble_count,
            "orientationScore": round(float(analysis["orientationScore"]), 4),
            "answerScore": round(float(analysis["answerScore"]), 4),
            "autoApprovalEligible": confidence >= settings.min_confidence_for_auto_approval
            and not should_retake
            and multiple_count == 0
            and low_confidence_count == 0,
        },
    )


def render_pdf_page(document: pdfium.PdfDocument, index: int) -> np.ndarray:
    page = document[index]
    try:
        bitmap = page.render(scale=settings.pdf_render_scale)
        pil_image = bitmap.to_pil().convert("RGB")
    finally:
        close_page = getattr(page, "close", None)
        if callable(close_page):
            close_page()

    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_rendered_pdf_page(
    index: int,
    page_count: int,
    page_image: np.ndarray,
    payload: OMRProcessPayload,
) -> tuple[int, OMRProcessResponse]:
    response = process_omr_array(page_image, payload, content_type="image/jpeg")
    response.metadata = {
        **(response.metadata or {}),
        "sourcePage": index + 1,
        "sourcePageCount": page_count,
        "pageConcurrency": settings.page_concurrency,
        "pdfRenderScale": settings.pdf_render_scale,
    }
    return index, response


def process_omr_file_pages(
    file_bytes: bytes,
    payload: OMRProcessPayload,
    content_type: str | None = None,
) -> list[OMRProcessResponse]:
    normalized_type = (content_type or "").split(";")[0].strip().lower()
    responses: list[OMRProcessResponse] = []

    if normalized_type == "application/pdf" or file_bytes.startswith(b"%PDF"):
        try:
            document = pdfium.PdfDocument(file_bytes)
        except Exception as error:
            raise ValueError("PDF_DECODE_FAILED") from error

        try:
            page_count = len(document)
            if page_count < 1:
                raise ValueError("PDF_EMPTY")
            if page_count > settings.max_pdf_pages:
                raise ValueError("PDF_PAGE_LIMIT_EXCEEDED")

            page_workers = min(settings.page_concurrency, page_count)
            if page_workers <= 1:
                for index in range(page_count):
                    page_image = render_pdf_page(document, index)
                    _, response = process_rendered_pdf_page(index, page_count, page_image, payload)
                    responses.append(response)
                    del page_image
                    gc.collect()
            else:
                responses_by_index: list[OMRProcessResponse | None] = [None] * page_count
                pending: dict[Future[tuple[int, OMRProcessResponse]], int] = {}

                def collect_finished_page() -> None:
                    future = next(as_completed(pending))
                    page_index, response = future.result()
                    responses_by_index[page_index] = response
                    pending.pop(future, None)
                    gc.collect()

                with ThreadPoolExecutor(max_workers=page_workers) as executor:
                    for index in range(page_count):
                        page_image = render_pdf_page(document, index)
                        future = executor.submit(process_rendered_pdf_page, index, page_count, page_image, payload)
                        pending[future] = index
                        del page_image

                        if len(pending) >= page_workers:
                            collect_finished_page()

                    while pending:
                        collect_finished_page()

                responses.extend(response for response in responses_by_index if response is not None)
        finally:
            close_document = getattr(document, "close", None)
            if callable(close_document):
                close_document()

        return responses

    page = decode_omr_file(file_bytes, content_type)
    response = process_omr_array(page, payload, content_type="image/jpeg")
    response.metadata = {
        **(response.metadata or {}),
        "sourcePage": 1,
        "sourcePageCount": 1,
        "pageConcurrency": 1,
    }
    responses.append(response)

    return responses
