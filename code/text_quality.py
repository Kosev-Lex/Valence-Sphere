# text_quality.py
import re

_BULLET_LINE = re.compile(r"^\s*[\-\*\u2022]\s+")
_NUM_LINE = re.compile(r"^\s*\d+\.\s+")

def clean_answer_text(text: str) -> str:
    if not text:
        return ""
    s = text.strip()

    # remove list bullets / numbered bullets at start of lines
    lines = []
    for ln in s.splitlines():
        ln = _BULLET_LINE.sub("", ln)
        ln = _NUM_LINE.sub("", ln)
        if ln.strip():
            lines.append(ln.strip())
    s = " ".join(lines)

    # strip stray trailing "1." artifact
    s = s.rstrip()
    s = re.sub(r"(?:\s*1\.\s*)+$", "", s).strip()

    # collapse whitespace and fix spaces before punctuation
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()

def is_incomplete(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    t = text.strip().lower()
    if t.endswith(":"):
        return True
    if "incomplete" in t or "retry recommended" in t:
        return True
    return False
