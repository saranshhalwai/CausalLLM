#!/usr/bin/env python3
"""
Simple PDF text extraction for the PS1 folder.
Saves per-PDF .txt files to PS1/text_outputs/ and a summaries.json.
Uses PyPDF2 first with a pdfminer.six fallback when PyPDF2 returns little or no text.
"""
import json
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

from pdfminer.high_level import extract_text


def extract_with_pypdf(path: Path):
    if PdfReader is None:
        return "", 0
    try:
        reader = PdfReader(str(path))
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                texts.append("")
        text = "\n".join(texts).strip()
        return text, len(reader.pages)
    except Exception:
        return "", 0


def extract_with_pdfminer(path: Path):
    try:
        t = extract_text(str(path))
        return t or ""
    except Exception:
        return ""


def main():
    base = Path(__file__).parent
    ps1 = base / "PS1"
    if not ps1.exists():
        print("PS1 folder not found at:", ps1)
        return
    outdir = ps1 / "text_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    pdfs = sorted(ps1.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in PS1 folder.")
        return

    for pdf in pdfs:
        print(f"Processing {pdf.name}...")
        text, pages = extract_with_pypdf(pdf)
        used = "pypdf"
        if not text or len(text) < 200:
            text2 = extract_with_pdfminer(pdf)
            if text2 and len(text2) > len(text):
                text = text2
                used = "pdfminer"

        if not text:
            print(f"Warning: no text extracted from {pdf.name}")

        outpath = outdir / (pdf.stem + ".txt")
        try:
            outpath.write_text(text, encoding="utf-8")
        except Exception as e:
            print("Failed to write text for", pdf.name, "->", e)

        summary = {
            "filename": pdf.name,
            "pages": pages,
            "extract_method": used,
            "chars": len(text),
            "words": len(text.split()),
            "preview": (text[:1000].replace("\n", " ") if text else "")
        }
        summaries.append(summary)

    summary_path = outdir / "summaries.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("Done. Outputs written to", str(outdir))
    for s in summaries:
        print(f"- {s['filename']}: pages={s['pages']}, words={s['words']}, method={s['extract_method']}")


if __name__ == "__main__":
    main()

