#!/usr/bin/env python3
"""Run lightweight repository checks without mutating artifacts."""

import compileall
from pathlib import Path


assert Path("README.md").is_file()
assert Path("docs/index.html").is_file()
assert Path("docs/Quantum_Radiation_Flow_APA_Report.pdf").is_file()
assert compileall.compile_dir("src", quiet=1)
assert compileall.compile_dir("tests", quiet=1)
print("PASS: repository structure and syntax")
