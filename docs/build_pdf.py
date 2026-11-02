"""Build docs/Project_Journey.pdf from the Markdown documentation set.

Pipeline: Markdown -> HTML (python-markdown) -> PDF (headless Chrome/Edge).

Usage:
    .\.venv\Scripts\python.exe docs\build_pdf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT_HTML = DOCS / "_journey.html"
OUT_PDF = DOCS / "Project_Journey.pdf"

DOC_FILES = [
    ("Complete Project Journey", "JOURNEY.md"),
    ("Decision Log", "DECISIONS.md"),
    ("Challenge Catalog", "CHALLENGES.md"),
    ("Experiments, Hyperparameters, and Tuning", "EXPERIMENTS.md"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Calibri", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #1b1f24;
  margin: 0;
}
h1 { font-size: 22pt; border-bottom: 2px solid #2f6feb; padding-bottom: 6px; margin: 0 0 14px; }
h2 { font-size: 16pt; margin-top: 26px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
h3 { font-size: 13pt; margin-top: 20px; }
h4 { font-size: 11.5pt; margin-top: 16px; }
p, ul, ol { margin: 8px 0; }
li { margin: 3px 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0;
  font-size: 9.2pt;
  page-break-inside: auto;
}
th, td { border: 1px solid #c9d1d9; padding: 5px 7px; vertical-align: top; }
th { background: #eef3fb; text-align: left; }
tr { page-break-inside: avoid; }
code {
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 9pt;
  background: #f3f5f8;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 5px;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 8.8pt;
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; }
blockquote {
  margin: 10px 0;
  padding: 8px 12px;
  border-left: 4px solid #2f6feb;
  background: #f6f9ff;
  color: #333;
}
hr { border: 0; border-top: 1px solid #d0d7de; margin: 22px 0; }
.cover {
  height: 245mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
  page-break-after: always;
}
.cover .title { font-size: 30pt; font-weight: 700; color: #123c7a; line-height: 1.25; }
.cover .subtitle { font-size: 14pt; color: #444; margin-top: 14px; }
.cover .meta { font-size: 11pt; color: #555; margin-top: 34px; line-height: 1.9; }
.doc { page-break-before: always; }
.doc:first-of-type { page-break-before: avoid; }
a { color: #2f6feb; text-decoration: none; }
"""


def find_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for name in ("chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def build_html() -> None:
    converter = markdown.Markdown(
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"]
    )

    sections = []
    for title, filename in DOC_FILES:
        path = DOCS / filename
        if not path.exists():
            raise SystemExit(f"Missing markdown file: {path}")
        body = converter.convert(path.read_text(encoding="utf-8"))
        converter.reset()
        sections.append(f'<section class="doc"><h1>{title}</h1>\n{body}</section>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Video Anomaly Detection - Project Journey</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <div class="title">Video Anomaly Detection<br>Complete Project Journey</div>
  <div class="subtitle">Decisions, challenges, experiments, hyperparameters, and tuning</div>
  <div class="meta">
    Dataset: ShanghaiTech Campus / CUHK Avenue / UCSD Ped<br>
    Framework: Anomalib 2.7.0, PyTorch 2.11.0+cu128<br>
    Hardware: NVIDIA RTX 3060 12 GB, Windows<br>
    Date: 2026-11-02
  </div>
</div>
{''.join(sections)}
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")


def build_pdf() -> None:
    browser = find_browser()
    if browser is None:
        raise SystemExit("No Chrome or Edge found; cannot print PDF")

    with tempfile.TemporaryDirectory(prefix="pdf-profile-") as profile:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={OUT_PDF}",
            OUT_HTML.resolve().as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"PDF generation failed (exit {result.returncode})")

    if not OUT_PDF.exists() or OUT_PDF.stat().st_size < 10_000:
        raise SystemExit("PDF was not produced or is suspiciously small")

    size = OUT_PDF.stat().st_size
    print(f"PDF written: {OUT_PDF} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_html()
    build_pdf()
