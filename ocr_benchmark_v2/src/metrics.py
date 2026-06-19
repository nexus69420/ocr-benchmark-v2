"""OCR evaluation metrics (character and word level)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EditCounts:
    distance: int
    insertions: int
    deletions: int
    substitutions: int


def _edit_counts(a: list[str], b: list[str]) -> EditCounts:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = m, n
    ins = dels = subs = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return EditCounts(distance=dp[m][n], insertions=ins, deletions=dels, substitutions=subs)


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    return _edit_counts(list(reference), list(hypothesis)).distance


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(_edit_counts(list(reference), list(hypothesis)).distance / len(reference), 6)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_counts(ref, hyp).distance / len(ref), 6)


def normalized_levenshtein_similarity(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 1.0
    denom = max(len(reference), len(hypothesis))
    if denom == 0:
        return 1.0
    ed = levenshtein_distance(reference, hypothesis)
    return round(1.0 - ed / denom, 6)


def anls(reference: str, hypothesis: str, threshold: float = 0.5) -> float:
    nl = normalized_levenshtein_similarity(reference, hypothesis)
    return round(nl if nl >= threshold else 0.0, 6)


def char_edit_details(reference: str, hypothesis: str) -> EditCounts:
    return _edit_counts(list(reference), list(hypothesis))


def word_edit_details(reference: str, hypothesis: str) -> EditCounts:
    return _edit_counts(reference.split(), hypothesis.split())


def length_ratio(reference: str, hypothesis: str) -> float | None:
    if not reference:
        return None
    return round(len(hypothesis) / len(reference), 6)
