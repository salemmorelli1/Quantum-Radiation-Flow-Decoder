#!/usr/bin/env python3
"""Run lightweight repository checks without mutating artifacts."""

import compileall
import json
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from quantum_radiation_flow import __version__


def main():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    pdf = PdfReader(root / "docs/Quantum_Radiation_Flow_APA_Report.pdf")
    assert len(pdf.pages) == 25, "the report must remain exactly 25 pages"
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() for page in pdf.pages))
    document = Document(root / "report/Quantum_Radiation_Flow_APA_Report.docx")
    docx_text = " ".join(p.text for p in document.paragraphs)
    for marker in (
        "NOT_ASSESSED",
        "heuristic interpolation",
        "seed-level intervals remain planned",
    ):
        assert marker in text, f"PDF missing current scientific scope: {marker}"
        assert marker in docx_text, f"DOCX missing current scientific scope: {marker}"
    for page in pdf.pages:
        assert abs(float(page.mediabox.width) - 612) < 1
        assert abs(float(page.mediabox.height) - 792) < 1
    docs = (root / "docs/index.html").read_text("utf-8")
    home = (root / "index.html").read_text("utf-8")
    assert (
        home.replace(
            'href="docs/Quantum_Radiation_Flow_APA_Report.pdf"',
            'href="Quantum_Radiation_Flow_APA_Report.pdf"',
        )
        == docs
    )
    assert "visibility ** 2" in docs
    assert "not a computed state overlap" in docs
    metrics = json.loads((root / "report/reference_metrics.json").read_text("utf-8"))
    assert metrics["page_time_fidelity_surrogate"] == 0.4375
    assert (
        metrics["seed_status"]
        == "analytic Page benchmark; illustrative entropy and fidelity proxies"
    )
    assert compileall.compile_dir(root / "src", quiet=1)
    assert compileall.compile_dir(root / "tests", quiet=1)
    print(
        f"Release {__version__} verified: 25-page report; illustrative site; scientific claims not assessed."
    )


if __name__ == "__main__":
    main()
