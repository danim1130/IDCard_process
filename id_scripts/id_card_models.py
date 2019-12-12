from typing import Tuple, List
from enum import Enum, auto


class Rectangle:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class IDCardFieldTypeEnum(Enum):
    TEXT = auto()
    TEXT_NAME = auto()
    TEXT_CITY = auto()
    TEXT_DATE = auto()
    BARCODE = auto()
    DATAGRAM = auto()


class IDCardFieldDescriptor:
    def __init__(self, key: str, rectangle: Rectangle, type: IDCardFieldTypeEnum):
        self.key = key
        self.rectangle = rectangle
        self.type = type


class IDCardConfiguration:
    def __init__(self, template, fields: List[IDCardFieldDescriptor], language: str) -> None:
        self.template = template
        self.fields = fields
        self.language = language


class ValidationField:
    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value


class ValidationResult:
    def __init__(self, key: str, is_correct: bool, confidence: int) -> None:
        self.key = key
        self.isCorrect = is_correct
        self.confidece = confidence

    def __str__(self) -> str:
        return f"key: {self.key}, isCorrect: {self.isCorrect}, confidence: {self.confidece}"


class ConfidenceValue:
    def __init__(self, value, confidence):
        self.value = value
        self.confidence = confidence
