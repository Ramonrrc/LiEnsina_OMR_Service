from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.schemas import OMRProcessPayload  # noqa: E402
from app.services.omr_processor import process_omr_file_pages  # noqa: E402


def build_payload(question_count: int) -> OMRProcessPayload:
    labels = ["A", "B", "C", "D", "E"]
    return OMRProcessPayload.model_validate(
        {
            "examId": "benchmark-exam",
            "versionId": f"benchmark-{question_count}",
            "classId": "benchmark-class",
            "skipQr": True,
            "answerKey": [
                {
                    "questionNumber": index + 1,
                    "questionId": f"question-{index + 1}",
                    "correctOption": labels[index % len(labels)],
                }
                for index in range(question_count)
            ],
        }
    )


def process_once(file_bytes: bytes, content_type: str, payload: OMRProcessPayload) -> dict[str, object]:
    start = time.perf_counter()
    responses = process_omr_file_pages(file_bytes, payload, content_type=content_type)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "elapsedMs": round(elapsed_ms, 2),
        "pages": len(responses),
        "avgConfidence": round(statistics.mean([response.confidence for response in responses]) if responses else 0, 4),
        "needsReview": sum(1 for response in responses if response.requiresReview),
        "retake": sum(1 for response in responses if response.shouldRetakeImage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local do OMR com answerKey de ate 100 questoes.")
    parser.add_argument("--file", required=True, help="Caminho da imagem ou PDF de cartao resposta.")
    parser.add_argument("--questions", type=int, default=100, help="Quantidade de questoes no gabarito.")
    parser.add_argument("--iterations", type=int, default=1, help="Total de execucoes.")
    parser.add_argument("--concurrency", type=int, default=1, help="Execucoes simultaneas.")
    args = parser.parse_args()

    if args.questions < 1 or args.questions > 100:
        raise SystemExit("--questions deve ficar entre 1 e 100.")
    if args.iterations < 1:
        raise SystemExit("--iterations deve ser maior que zero.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency deve ser maior que zero.")

    file_path = Path(args.file).expanduser().resolve()
    file_bytes = file_path.read_bytes()
    suffix = file_path.suffix.lower()
    content_type = "application/pdf" if suffix == ".pdf" else "image/jpeg"
    payload = build_payload(args.questions)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.iterations)) as executor:
        futures = [
            executor.submit(process_once, file_bytes, content_type, payload)
            for _ in range(args.iterations)
        ]
        results = [future.result() for future in as_completed(futures)]
    total_ms = (time.perf_counter() - started) * 1000
    elapsed = [float(result["elapsedMs"]) for result in results]

    print(json.dumps({
        "file": str(file_path),
        "fileBytes": len(file_bytes),
        "settings": {
            "detectedCpuCores": settings.detected_cpu_cores,
            "requestConcurrency": settings.request_concurrency,
            "pageConcurrency": settings.page_concurrency,
            "opencvThreads": settings.opencv_threads,
            "pdfRenderScale": settings.pdf_render_scale,
            "maxPdfPages": settings.max_pdf_pages,
        },
        "config": {
            "questions": args.questions,
            "iterations": args.iterations,
            "concurrency": args.concurrency,
        },
        "timingsMs": {
            "min": round(min(elapsed), 2),
            "avg": round(statistics.mean(elapsed), 2),
            "max": round(max(elapsed), 2),
            "wallClock": round(total_ms, 2),
        },
        "results": results,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
