# Generated artifacts

`run_research_pipeline.py` writes versionable summaries to `artifacts/latest/`:

- `experiment_summary.json`
- `page_curve.csv`

Generated run directories are ignored by Git. Commit only explicitly frozen,
audited results used by a manuscript or release.
