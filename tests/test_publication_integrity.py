import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_current_report_site_and_scope_are_verified():
    subprocess.run([sys.executable, "scripts/part4_validate_repository.py"], cwd=ROOT, check=True)


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is required for browser runtime checks"
)
def test_actual_browser_math_matches_page_formula_and_squared_visibility_proxy(tmp_path):
    source = (ROOT / "docs/index.html").read_text("utf-8")
    script = source.split("<script>", 1)[1].split("</script>", 1)[0]
    file = tmp_path / "site.js"
    file.write_text(script, encoding="utf-8")
    subprocess.run(["node", "--check", str(file)], check=True, capture_output=True)
    functions = []
    for name in ("harmonic", "pageEntropyAt", "baseFidelityAt"):
        match = re.search(rf"function {name}\([^)]*\)\s*\{{", script)
        assert match
        start = match.start()
        end = match.end()
        depth = 1
        while depth:
            depth += (script[end] == "{") - (script[end] == "}")
            end += 1
        functions.append(script[start:end])
    file.write_text(
        "const state={width:0.075};const sigmoid=x=>1/(1+Math.exp(-x));\n"
        + "\n".join(functions)
        + "\nconsole.log(JSON.stringify({page:pageEntropyAt(5,10),start:pageEntropyAt(0,10),mid:baseFidelityAt(0.5),late:baseFidelityAt(0.9)}));\n",
        encoding="utf-8",
    )
    output = subprocess.run(["node", str(file)], check=True, capture_output=True, text=True)
    data = json.loads(output.stdout)
    metrics = json.loads((ROOT / "report/reference_metrics.json").read_text())
    assert data["page"] == pytest.approx(metrics["page_peak_entropy_nats"], abs=1e-12)
    assert abs(data["start"]) < 1e-12
    assert data["mid"] == 0.4375
    assert math.isclose(data["late"], metrics["late_fidelity_surrogate"], abs_tol=1e-12)
