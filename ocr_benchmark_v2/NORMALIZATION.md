# Text normalization (v2 benchmark)

Applied identically to **reference** and **hypothesis** before metrics.

## Steps

1. **Unicode NFC** — canonical composed form for Gujarati.
2. **Strip BOM / zero-width** — remove `\ufeff`, `\u200b`, etc.
3. **Preserve newlines** — `\n` kept as line boundaries.
4. **Collapse horizontal whitespace** — runs of spaces/tabs within a line → single space; trim line ends.
5. **Mistral only (pre-normalize)** — strip markdown/HTML to plain text before step 1 (see `strip_markdown_for_scoring`).

## Not applied

- Case folding
- Gujarati spelling correction
- Digit normalization (Gujarati ↔ ASCII numerals)
- Table structure reconstruction

## Truncation heuristics (flags only)

- `output_truncated`: API `finish_reason=max_tokens`, or output ends mid-token (`[0-9.]+\s*$`), or length &lt; 50% of peer median on same page.
- `reference_truncated`: same heuristics on reference build.

## Scoring reliability

`scoring_reliable = false` when `reference_truncated` or `reference_empty`.
