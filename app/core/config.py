from dataclasses import dataclass
import os


def _read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return None


def _detect_cpu_cores() -> float:
    cpu_max = _read_text_file("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max":
            try:
                quota_value = float(quota)
                period_value = float(period)
                if quota_value > 0 and period_value > 0:
                    return max(0.5, quota_value / period_value)
            except ValueError:
                pass

    quota = _read_text_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_text_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period:
        try:
            quota_value = float(quota)
            period_value = float(period)
            if quota_value > 0 and period_value > 0:
                return max(0.5, quota_value / period_value)
        except ValueError:
            pass

    return float(os.cpu_count() or 2)


def _default_page_concurrency(cpu_cores: float) -> int:
    if cpu_cores < 2:
        return 1
    if cpu_cores < 3:
        return 2
    if cpu_cores < 5:
        return 3
    return 4


def _read_int_env(name: str, fallback: int, min_value: int, max_value: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback

    return max(min_value, min(max_value, value))


def _read_auto_int_env(name: str, fallback: int, min_value: int, max_value: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip().lower() in {"", "auto"}:
        return max(min_value, min(max_value, fallback))

    return _read_int_env(name, fallback, min_value, max_value)


def _read_float_env(name: str, fallback: float, min_value: float, max_value: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback

    return max(min_value, min(max_value, value))


_DETECTED_CPU_CORES = _detect_cpu_cores()
_DEFAULT_PAGE_CONCURRENCY = _default_page_concurrency(_DETECTED_CPU_CORES)


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("OMR_ENV", "development")
    internal_token: str = os.getenv("OMR_INTERNAL_TOKEN", "").strip()
    max_upload_mb: int = _read_int_env("OMR_MAX_UPLOAD_MB", 16, 1, 512)
    max_payload_kb: int = _read_int_env("OMR_MAX_PAYLOAD_KB", 128, 4, 1024)
    request_concurrency: int = _read_int_env("OMR_REQUEST_CONCURRENCY", 1, 1, 8)
    detected_cpu_cores: float = round(_DETECTED_CPU_CORES, 2)
    page_concurrency: int = _read_auto_int_env("OMR_PAGE_CONCURRENCY", _DEFAULT_PAGE_CONCURRENCY, 1, 6)
    opencv_threads: int = _read_int_env("OMR_OPENCV_THREADS", 1, 1, 8)
    pdf_render_scale: float = _read_float_env("OMR_PDF_RENDER_SCALE", 2.0, 1.5, 3.0)
    max_pdf_pages: int = _read_int_env("OMR_MAX_PDF_PAGES", 40, 1, 40)
    request_timeout_seconds: int = _read_int_env("OMR_REQUEST_TIMEOUT_SECONDS", 90, 1, 120)
    min_confidence_for_auto_approval: float = float(
        os.getenv("OMR_MIN_CONFIDENCE_FOR_AUTO_APPROVAL", "0.88")
    )

    def __post_init__(self) -> None:
        if self.env == "production" and len(self.internal_token) < 64:
            raise RuntimeError("OMR_INTERNAL_TOKEN must have at least 64 characters in production.")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_payload_bytes(self) -> int:
        return self.max_payload_kb * 1024


settings = Settings()
