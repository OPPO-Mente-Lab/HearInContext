# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HearInContext text normalization with process-local bounded caches."""
import json
import re
import string
from functools import lru_cache
from itertools import groupby
from pathlib import Path
import regex
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

POLICY = "hearincontext"
CACHE_SIZE = 65536
_ZH_NORMALIZER = None
_ASCII_DIGIT = re.compile(r"[0-9]")
_DIGIT = re.compile(r"\d")

# Fixed punctuation alphabet: ASCII punctuation and the Chinese/full-width
# separators accepted by the benchmark. Other symbols remain literal.
_SEPARATORS = string.punctuation + (
    "，；､、丶｟｠《》（）｢｣［］｛｝「『』【】〔〕〖〗〘〙〚〛〈〉｜："
    "！？｡。＂＃＄％＆＇＊＋－／＜＝＞＠＼＾＿｀～〃〜〝〞〟〰〾〿"
    "‘’‛“”„‟…‧﹏·•・′″–—―"
)
_SEPARATOR_TABLE = str.maketrans({char: " " for char in _SEPARATORS})
_SCRIPT = regex.compile(
    r"(?P<cjk>[\p{Han}\p{Hangul}\p{Hiragana}\p{Katakana}])|(?P<latin>\p{Latin})"
)


@lru_cache(maxsize=4096)
def _script_kind(char):
    match = _SCRIPT.fullmatch(char)
    return match.lastgroup if match else None


def _letter_unit(token):
    return (len(token) == 1 or (len(token) == 2 and token.endswith("s"))) and (
        token[0].isupper() or token[0].islower()
    )


def normalize_chinese_text(text):
    """Apply punctuation, script-boundary and letter-run rules locally.

    Chinese-family characters split from each other and adjacent Latin letters.
    Digits, combining marks and other symbols do not create script boundaries.
    Consecutive single cased letters (including an attached plural s) form one
    word token. Case folding follows segmentation, except all-uppercase inputs.
    """
    cleaned = (text.lower() if text.isupper() else text).translate(_SEPARATOR_TABLE)
    pieces = []
    previous = None
    for char in cleaned:
        current = _script_kind(char)
        if (previous == "cjk" and current in ("cjk", "latin")) or (
            previous == "latin" and current == "cjk"
        ):
            pieces.append(" ")
        pieces.append(char)
        previous = current
    words = []
    for join_letters, run in groupby("".join(pieces).split(), key=_letter_unit):
        words.extend(["".join(run)] if join_letters else run)
    return " ".join(words).lower()

@lru_cache(maxsize=32768)
def chinese_tn(text):
    """Fixed public TN; a digit-free string is returned byte-for-byte unchanged."""
    if not _ASCII_DIGIT.search(text):
        return text
    global _ZH_NORMALIZER
    if _ZH_NORMALIZER is None:
        from wetext import Normalizer
        _ZH_NORMALIZER = Normalizer(lang="zh", operator="tn")
    return _ZH_NORMALIZER.normalize(text)

_START = r"[A-Zb-hj-rt-z]"
_SUFFIX = r"(?:['’]?s)?"
_DOTTED = rf"{_START}\.[ \t]*(?:[A-Z]\.[ \t]*)*[A-Z]{_SUFFIX}(?!\w)\.?"
_SPACED = rf"{_START}(?:[ \t]+[A-Z])+{_SUFFIX}(?!\w)"
# One substitution chooses the leftmost whole run, so the final letter of a
# spaced initialism cannot start a second dotted match across a sentence end.
_INITIALISMS = re.compile(rf"(?<![\w'’])(?:{_DOTTED}|{_SPACED})")
_SPELLING = json.loads(Path(__file__).with_name("resources").joinpath("whisper_spelling.json").read_text(encoding="utf-8"))
_NORMALIZER = EnglishTextNormalizer(_SPELLING)


def merge_initialisms(text):
    def merge(match):
        value = match.group()
        if value.split() in (["I", "A"], ["A", "I"]):
            return value
        return re.sub(r"[. \t]", "", value)
    return _INITIALISMS.sub(merge, text)


@lru_cache(maxsize=32768)
def normalize_english(text):
    return _NORMALIZER(merge_initialisms(text))


@lru_cache(maxsize=CACHE_SIZE)
def _tokens(language, text):
    normalized = (normalize_chinese_text(chinese_tn(text)) if language == "zh"
                  else normalize_english(text))
    return tuple(normalized.split()), bool((_ASCII_DIGIT if language == "zh" else _DIGIT).search(text))


def normalize_pair(reference, hypothesis, language, targets=()):
    if language not in ("zh", "en"):
        raise ValueError("language must be zh or en")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("reference and hypothesis must be strings")
    values = [_tokens(language, text) for text in [reference, hypothesis, *targets]]
    return {"reference_tokens": list(values[0][0]),
            "hypothesis_tokens": list(values[1][0]),
            "target_tokens": [list(value[0]) for value in values[2:]],
            "digit_triggered": any(value[1] for value in values)}


normalize_pair.policy = POLICY




def contains_tokens(sequence, target):
    """One labeled target succeeds if an exact normalized token span exists."""
    width = len(target)
    return bool(width) and any(sequence[i:i + width] == target for i in range(len(sequence) - width + 1))


def clear_caches():
    """Clear process-local normalized text caches; leave rules unchanged."""
    for cache in (_tokens, chinese_tn, normalize_english):
        cache.cache_clear()
