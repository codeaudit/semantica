# PR Title
Harden spaCy NER Fallback in Semantic Extract

## Summary
This PR fixes a runtime failure in Semantic Extract when spaCy is installed but not actually usable at runtime, such as Python 3.12 / Pydantic v2 environments where `spacy.load("en_core_web_sm")` fails during config validation.

Instead of crashing during `NERExtractor(method="ml")` initialization or ML entity extraction, Semantica now logs the failure and falls back cleanly to non-ML extraction behavior.

## What Changed

### spaCy Initialization Hardening
Updated `semantica/semantic_extract/ner_extractor.py` so `NERExtractor` no longer crashes if:

- spaCy is importable
- the configured model exists
- but `spacy.load(...)` fails at runtime for reasons other than missing files

This now degrades gracefully by leaving `self.nlp = None` and allowing fallback behavior.

### ML Extraction Fallback Hardening
Updated `semantica/semantic_extract/methods.py` so `extract_entities_ml()` now catches:

- missing spaCy model errors
- generic spaCy runtime initialization failures

If spaCy cannot initialize, extraction falls back to pattern-based extraction instead of raising.

### Regression Test
Added a regression test in `tests/test_ner_configurations.py` covering the case where:

- spaCy is available
- `spacy.load(...)` raises a runtime exception
- `NERExtractor(method="ml")` still initializes safely

### Changelog
Added an `Unreleased` changelog entry documenting the spaCy runtime fallback fix.

## Why This Matters
This fixes benchmark and CI instability caused by spaCy runtime incompatibilities outside Semantica’s control.

It ensures Semantic Extract remains resilient when spaCy is present in the environment but broken due to dependency mismatches.

## Validation
Tested with:

```bash
pytest tests/test_ner_configurations.py -q -k "spacy_runtime_is_broken"
```

Result:

```bash
1 passed
```

Note:
The benchmark failure path was fixed directly, but the local `semantic-extract` branch did not contain the benchmark file path used in CI, so only the targeted regression path was verified locally.

## Files Changed

- `semantica/semantic_extract/ner_extractor.py`
- `semantica/semantic_extract/methods.py`
- `tests/test_ner_configurations.py`
- `CHANGELOG.md`
