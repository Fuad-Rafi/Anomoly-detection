"""Download arXiv PDFs for the literature review.

Reads a manifest of papers, resolves arXiv IDs or searches the arXiv API by title,
and downloads PDFs into `pdfs/`. Writes `manifest.md` with status for every entry.
"""

from __future__ import annotations

import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "vad-lit-review/1.0 (research use; contact: local)"}

# (slug, arxiv_id_or_None, title_for_search_or_None)
PAPERS = [
    ("liu2018future_frame_prediction", "1712.09867", None),
    ("gong2019_memae", "1904.02623", None),
    ("park2020_mnad", "2001.10921", None),
    ("tian2021_rtfm", "2101.10030", None),
    ("jeong2023_winclip", "2303.14814", None),
    ("zhou2024_anomalyclip", "2308.11564", None),
    ("roth2022_patchcore", "2106.08265", None),
    ("defard2021_padim", "2011.08785", None),
    ("yu2021_fastflow", "2111.07677", None),
    ("gudovskiy2022_cflow_ad", "2107.12571", None),
    ("wu2020_xd_violence", "2005.06442", None),
    ("zhong2019_graph_label_noise", "1903.02138", None),
    ("sultani2018_ucf_crime", "1801.04264", None),
    ("reiss2024_arvad", "2212.00789", None),
    ("reiss2023_mscl", None, "Mean-Shifted Contrastive Loss for Anomaly Detection"),
    ("wu2024_vadclip", None, "VadCLIP Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection"),
    ("georgescu2021_background_agnostic", None, "A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video"),
    ("georgescu2021_selfsupervised", None, "Anomaly Detection in Video via Self-Supervised and Multi-Task Learning"),
    ("ramachandra2020_survey", None, "A Survey of Single-Scene Video Anomaly Detection"),
    ("feng2021_mist", None, "MIST Multiple Instance Self-Training Framework for Video Anomaly Detection"),
]

# Pre-arXiv / paywalled classics: no public PDF, listed in the manifest only.
NO_PDF = [
    ("kim2009_observe_locally", "CVPR 2009", "https://grauman.cs.utexas.edu/projects/accv2010/"),
    ("mahadevan2010_mdt", "CVPR 2010", "https://www.svcl.ucsd.edu/projects/anomaly/"),
    ("cong2011_sparse", "CVPR 2011", None),
    ("zhao2011_dynamic_sparse", "CVPR 2011", None),
    ("lu2013_150fps", "ICCV 2013", "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/"),
    ("reiss2024_aivad", "CVPR 2024 (open access)", "https://openaccess.thecvf.com/content/CVPR2024/html/Reiss_AI-VAD_Accurate_and_Interpretable_Video_Anomaly_Detection_CVPR_2024_paper.html"),
    ("georgescu2022_ssagan", "IEEE TNNLS 2022", "https://ieeexplore.ieee.org/document/9794848"),
    ("liu2020_clustering_autoencoder", "ECCV 2020", "https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2931_ECCV_2020_paper.php"),
]


def search_arxiv(title: str) -> dict | None:
    query = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=1"
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title_el = entry.find("a:title", ns)
    link_el = entry.find("a:link[@title='pdf']", ns)
    if link_el is None:
        link_el = entry.find("a:link", ns)
    resolved_title = title_el.text.strip() if title_el is not None else title
    pdf_url = link_el.get("href") if link_el is not None else None
    return {"title": resolved_title, "pdf": pdf_url}


def download(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
    response = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    if response.status_code != 200:
        return False
    if "html" in response.headers.get("Content-Type", ""):
        return False
    with out_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            handle.write(chunk)
    return out_path.exists() and out_path.stat().st_size > 1000


def main() -> None:
    manifest_lines = [
        "# Literature review — PDF manifest",
        "",
        "| File | Source | Status | Resolved title / link |",
        "|---|---|---|---|",
    ]

    for slug, arxiv_id, title in PAPERS:
        out_path = PDF_DIR / f"{slug}.pdf"
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            ok = download(pdf_url, out_path)
            manifest_lines.append(
                f"| `{slug}.pdf` | arXiv:{arxiv_id} | {'OK' if ok else 'FAILED'} | "
                f"{pdf_url} |"
            )
            print(f"{'OK ' if ok else 'FAIL'} {slug}  (arXiv:{arxiv_id})")
            time.sleep(1)
            continue

        found = search_arxiv(title)
        if found and found["pdf"]:
            ok = download(found["pdf"], out_path)
            manifest_lines.append(
                f"| `{slug}.pdf` | arXiv search | {'OK' if ok else 'FAILED'} | "
                f"{found['title']} — {found['pdf']} |"
            )
            print(f"{'OK ' if ok else 'FAIL'} {slug}  -> {found['title']}")
        else:
            manifest_lines.append(
                f"| `{slug}` | arXiv search | NOT FOUND | {title} |"
            )
            print(f"MISS {slug}  ({title})")
        time.sleep(3)

    for slug, venue, link in NO_PDF:
        manifest_lines.append(
            f"| `{slug}` | {venue} | PAYWALLED | {link or 'no public PDF'} |"
        )

    manifest_lines.append("")
    manifest_lines.append("> Regenerate with `python download_papers.py`.")
    (HERE / "manifest.md").write_text("\n".join(manifest_lines), encoding="utf-8")
    print("\nmanifest.md written")


if __name__ == "__main__":
    main()
