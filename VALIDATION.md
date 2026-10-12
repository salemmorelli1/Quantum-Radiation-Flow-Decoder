# Validation Record

## October 12 2026 audit

Release 0.2.0 was verified using the installed CPU wheel on Python 3.12:
88 tests, Ruff, strict mypy, compilation, dependency checks, CLI execution,
model-checkpoint reload, and artifact fingerprints passed. Node checked the
actual browser math and JavaScript syntax. The corrected report was rendered,
all 25 pages inspected, and its scientific scope verified automatically.

The full 1,200-step default single-seed run completed with late decoder fidelity
0.208128 versus independent-null fidelity 0.251103. Scientific validation remains
`NOT_ASSESSED`; software verification does not certify a successful decoder.
See [the current audit](docs/DENSITY_RECOVERY_AUDIT_2026-10-12.md) and its evidence.
The CI matrix now checks built wheels on Linux/Windows and Python 3.11/3.12.

## Historical packaging record

Validation performed on August 24, 2026:

- Python source, tests, scripts, and report-reproduction code compile successfully.
- `docs/index.html` parses successfully; every local link resolves.
- Embedded JavaScript passes `node --check` after extraction from the HTML.
- The linked report is a tagged, letter-sized PDF with exactly 25 pages.
- The editable DOCX passed the document accessibility audit with zero high-,
  medium-, or low-severity findings.
- All five DOCX tables have internally consistent fixed geometry and repeating
  header rows.
- Every report page was rendered and visually inspected for clipping, overflow,
  and missing content.

PyTorch was unavailable in the packaging container, so the unit-test suite was
not represented as locally executed. The included GitHub Actions workflow installs
the declared dependencies and runs Ruff, Pytest, and compilation on Python 3.11
and 3.12 after the repository is pushed.

