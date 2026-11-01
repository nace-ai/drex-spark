"""The shared SQL surface's rules: criteria and questions, the checks made before any request, and answers as SQL
values.

They copy drex-duckdb's `prompt_drex` and `drex_decide` (src/drex_functions.cpp), so a question that one refuses the
other refuses too. Pure Python, so the PySpark functions and the Unity Catalog handlers share them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "drex-v2.0"
#: Only drex_decide_image defaults to an image model; the typed helpers never switch to one by themselves
DEFAULT_IMAGE_MODEL = "drex-max"
#: Drex's limits: questions per request, and labels per choice or levels per score
MAX_QUESTIONS = 512
MIN_LABELS = 2
MAX_LABELS = 255
MAX_IMAGES = 8

QUESTION_TYPES = ("noul", "choice", "score")


class DrexError(RuntimeError):
    """A failure that fails the query: a refused question, an unknown model, a 401, a 402. It names the function."""


def fail(function: str, message: str) -> DrexError:
    return DrexError(f"{function}: {message}")


@dataclass(frozen=True)
class Label:
    label: str
    #: `None` when the label has no description, which Drex reads differently from an empty one
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    labels: tuple[Label, ...]
    #: Whether the labels were written as Drex's object of label to description, where a noul may describe only one
    labels_from_object: bool = False

    def to_wire(self) -> dict[str, Any]:
        """The question as Drex takes it, with `instructions` always a string."""
        wire: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.type == "score":
            wire["criteria"] = [label.label for label in self.labels]
        elif self.type == "choice" or any(label.description is not None for label in self.labels):
            # Noul labels alone, with no description, are the same as no criteria
            wire["criteria"] = {label.label: label.description for label in self.labels}
        return wire


def _json(function: str, name: str, value: Any) -> Any:
    """A JSON text argument parsed, or a Python value as given."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError as error:
        raise fail(function, f"{name} is not valid JSON: {error}") from None


def _text_or_none(value: Any, shape_error: DrexError) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise shape_error
    return value


def _label(function: str, name: str, label: str | None, description: str | None) -> Label:
    if label is None:
        raise fail(function, f"{name} must not contain a NULL label")
    return Label(label, description)


def parse_labels(function: str, name: str, value: Any, *, allow_object: bool) -> tuple[tuple[Label, ...], bool]:
    """The labels of `criteria`, `labels` or `levels`, and whether they were written as an object.

    Accepted: a list of labels, a list of `{"label", "description"}`, or (for a noul or a choice) an object of label to
    description, as JSON text or the Python value. A NumPy array, which is how Spark hands over an ARRAY, is a list.
    """
    value = _json(function, name, value)
    shape_error = fail(
        function,
        f"{name} must be a list of labels or of {{label, description}} objects"
        + (", or an object of label to description" if allow_object else ""),
    )
    if hasattr(value, "tolist") and not isinstance(value, Mapping):
        value = value.tolist()
    if isinstance(value, Mapping):
        if not allow_object:
            raise shape_error
        return tuple(_label(function, name, key, _text_or_none(text, shape_error)) for key, text in value.items()), True
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise shape_error
    labels = []
    for element in value:
        if isinstance(element, Mapping):
            if set(element) - {"label", "description"}:
                raise shape_error
            labels.append(
                _label(
                    function,
                    name,
                    _text_or_none(element.get("label"), shape_error),
                    _text_or_none(element.get("description"), shape_error),
                )
            )
        else:
            labels.append(_label(function, name, _text_or_none(element, shape_error), None))
    return tuple(labels), False


def _check_labels(function: str, name: str, question: Question) -> None:
    labels = question.labels
    if question.type == "noul":
        # No labels is a noul without criteria
        names = [label.label for label in labels]
        valid = all(label in ("true", "false") for label in names) and len(set(names)) == len(names)
        complete = not labels or question.labels_from_object or set(names) == {"true", "false"}
        if not valid or not complete:
            raise fail(function, f"{name} labels must be exactly true and false, got [{', '.join(names)}]")
        return
    kind = "labels" if question.type == "choice" else "levels"
    if not MIN_LABELS <= len(labels) <= MAX_LABELS:
        raise fail(function, f"{name} takes {MIN_LABELS} to {MAX_LABELS} {kind}, got {len(labels)}")
    seen: set[str] = set()
    for label in labels:
        if not label.label:
            raise fail(function, f"{name} {kind} must not be empty")
        if label.label in seen:
            raise fail(function, f"{name} {kind} must be unique; '{label.label}' is a duplicate")
        seen.add(label.label)
        if question.type == "score" and label.description is not None:
            raise fail(function, "score levels take no description; describe them in instructions")


def _check_noul_has_text(function: str, name: str, question: Question) -> None:
    """What Drex would refuse in a noul: one with neither instructions nor a description of true or false."""
    if question.type != "noul" or question.instructions or any(label.description for label in question.labels):
        return
    raise fail(function, f"{name} needs instructions, or a description of true or false")


def _question_type(function: str, name: str, value: Any) -> str:
    if value in QUESTION_TYPES:
        return value
    if value in ("multiple_choice", "box"):
        raise fail(function, f"{name}: {value} questions are not supported, since Drex has no {value} question")
    if not isinstance(value, str):
        raise fail(function, f"{name}.type must be noul, choice or score")
    raise fail(function, f"{name}.type must be noul, choice or score, not '{value}'")


def _check_questions(function: str, questions: Sequence[Question], *, named: bool) -> None:
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise fail(function, f"questions takes 1 to {MAX_QUESTIONS} questions, got {len(questions)}")
    ids: set[str] = set()
    for question in questions:
        name = f"questions.{question.id}" if named else question.id
        if not question.id.strip():
            raise fail(function, "question ids must not be empty")
        # Drex collects answers in a plain object, whose __proto__ is not a key, so it would bill the request and drop
        # the answer
        if question.id == "__proto__":
            raise fail(function, "the question id '__proto__' is not allowed: Drex can't answer it")
        if question.id.lower() in ids:
            raise fail(function, f"question ids must be unique ignoring case; '{question.id}' is a duplicate")
        ids.add(question.id.lower())
        _check_labels(function, name, question)
        _check_noul_has_text(function, name, question)


def typed_question(function: str, kind: str, instructions: str | None, criteria: Any) -> Question:
    """The one question of `drex_noul`, `drex_choice` or `drex_score`, checked. NULL or '' instructions are omitted."""
    labels: tuple[Label, ...] = ()
    from_object = False
    if criteria is not None:
        argument = {"noul": "criteria", "choice": "labels", "score": "levels"}[kind]
        labels, from_object = parse_labels(function, argument, criteria, allow_object=kind != "score")
    elif kind != "noul":
        raise fail(function, f"{kind} needs {'labels' if kind == 'choice' else 'levels'}")
    if instructions is not None and not isinstance(instructions, str):
        raise fail(function, "instructions must be a string")
    question = Question(kind, kind, instructions or "", labels, from_object)
    _check_questions(function, [question], named=False)
    return question


def parse_questions(function: str, value: Any) -> tuple[Question, ...]:
    """The questions of `drex_decide` and `drex_decide_image`: an object of id to `{type, instructions, criteria}`."""
    value = _json(function, "questions", value)
    if not isinstance(value, Mapping):
        raise fail(function, "questions must be a JSON object of question id to question")
    questions = []
    for qid, entry in value.items():
        name = f"questions.{qid}"
        if not isinstance(entry, Mapping):
            raise fail(function, f"{name} must be an object with type, instructions and criteria")
        for key in entry:
            if key not in ("type", "instructions", "criteria"):
                raise fail(function, f"{name} has an unknown key '{key}'; use type, instructions and criteria")
        kind = _question_type(function, name, entry.get("type"))
        instructions = entry.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise fail(function, f"{name}.instructions must be a string")
        criteria = entry.get("criteria")
        labels: tuple[Label, ...] = ()
        from_object = False
        if criteria is not None:
            labels, from_object = parse_labels(function, f"{name}.criteria", criteria, allow_object=kind != "score")
        elif kind != "noul":
            raise fail(function, f"{name} needs criteria")
        questions.append(Question(str(qid), kind, instructions or "", labels, from_object))
    _check_questions(function, questions, named=True)
    return tuple(questions)


def wire_questions(questions: Sequence[Question]) -> dict[str, Any]:
    return {question.id: question.to_wire() for question in questions}


# ---------------------------------------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------------------------------------


def _number(function: str, question: Question, answer: Any, key: str) -> float:
    value = answer.get(key) if isinstance(answer, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DrexError(f"{function}: Drex's answer to question '{question.id}' has no number {key}")
    return float(value)


def answer_value(function: str, question: Question, answers: Any) -> Any:
    """Drex's answer to the question as its SQL value: a noul's probability, or a choice or score row.

    A choice's probabilities are sorted by label, a score's by index, as drex-duckdb returns them. Values come from the
    response and are never invented: an answer missing a field fails the query.
    """
    answer = answers.get(question.id) if isinstance(answers, Mapping) else None
    if not isinstance(answer, Mapping):
        raise DrexError(f"{function}: Drex's response has no answer to question '{question.id}'")
    if question.type == "noul":
        return _number(function, question, answer, "noul")
    probabilities = answer.get("probabilities")
    confidence = _number(function, question, answer, "confidence")
    if question.type == "choice":
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise DrexError(f"{function}: Drex's answer to question '{question.id}' has no choice")
        rows = [
            {"value": label, "probability": _number(function, question, probabilities, label)}
            for label in sorted(label.label for label in question.labels)
        ]
        return {"choice": choice, "probabilities": rows, "confidence": confidence}
    rows = [
        {"index": index, "value": label.label, "probability": _number(function, question, probabilities, str(index))}
        for index, label in enumerate(question.labels)
    ]
    return {"score": _number(function, question, answer, "score"), "probabilities": rows, "confidence": confidence}
