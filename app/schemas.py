from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


OptionLabel = Literal["A", "B", "C", "D", "E"]
QuestionStatus = Literal["ok", "blank", "multiple", "low_confidence", "unreadable"]


class AnswerKeyItem(BaseModel):
    questionNumber: int = Field(..., ge=1)
    correctOption: OptionLabel
    questionId: str | None = None


class OMRProcessPayload(BaseModel):
    examId: str
    versionId: str
    answerCardId: str | None = None
    studentId: str | None = None
    classId: str | None = None
    templateVersion: str = "liensina-omr-v1"
    options: list[OptionLabel] = ["A", "B", "C", "D", "E"]
    answerKey: list[AnswerKeyItem]

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
