from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("OMR_ENV", "development")
    max_upload_mb: int = int(os.getenv("OMR_MAX_UPLOAD_MB", "16"))
    min_confidence_for_auto_approval: float = float(
        os.getenv("OMR_MIN_CONFIDENCE_FOR_AUTO_APPROVAL", "0.88")
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
