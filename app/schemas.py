from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


OptionLabel = Literal["A", "B", "C", "D", "E"]
QuestionStatus = Literal["ok", "blank", "multiple", "low_confidence", "unreadable"]


class AnswerKeyItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questionNumber: int = Field(..., ge=1)
    correctOption: OptionLabel
    questionId: str | None = Field(default=None, max_length=128)


class OMRProcessPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examId: str = Field(..., min_length=1, max_length=128)
    versionId: str = Field(..., min_length=1, max_length=128)
    answerCardId: str | None = Field(default=None, max_length=128)
    studentId: str | None = Field(default=None, max_length=128)
    classId: str | None = Field(default=None, max_length=128)
    templateVersion: str = Field(default="liensina-omr-v1", max_length=64)
    skipQr: bool = False
    options: list[OptionLabel] = Field(default_factory=lambda: ["A", "B", "C", "D", "E"], min_length=2, max_length=5)
    answerKey: list[AnswerKeyItem] = Field(..., min_length=1, max_length=180)

    @field_validator("answerKey")
    @classmethod
    def answer_key_must_be_contiguous(cls, value: list[AnswerKeyItem]) -> list[AnswerKeyItem]:
        numbers = sorted(item.questionNumber for item in value)
        expected = list(range(1, len(value) + 1))
        if numbers != expected:
            raise ValueError("answerKey deve conter questionNumber sequencial iniciando em 1.")
        return value


class QualityReport(BaseModel):
    width: int
    height: int
    brightness: float
    contrast: float
    blur: float
    warnings: list[str]
    confidence: float


class QRCodeReport(BaseModel):
    found: bool
    raw: str | None = None
    parsed: dict[str, Any] | None = None
    warnings: list[str] = []


class BubbleOptionScore(BaseModel):
    option: OptionLabel
    fillRatio: float


class DetectedAnswer(BaseModel):
    questionNumber: int
    questionId: str | None = None
    detectedOption: OptionLabel | None
    correctOption: OptionLabel
    isCorrect: bool
    status: QuestionStatus
    confidence: float
    markedOptions: list[OptionLabel]
    optionScores: list[BubbleOptionScore]


class OMRProcessResponse(BaseModel):
    examId: str
    versionId: str
    answerCardId: str | None
    studentId: str | None
    classId: str | None
    templateVersion: str
    suggestedScore: float
    correctCount: int
    wrongCount: int
    blankCount: int
    multipleCount: int
    totalQuestions: int
    confidence: float
    requiresReview: bool
    shouldRetakeImage: bool
    failures: list[str]
    quality: QualityReport
    qrCode: QRCodeReport
    detectedAnswers: list[DetectedAnswer]
    metadata: dict[str, Any] = {}
