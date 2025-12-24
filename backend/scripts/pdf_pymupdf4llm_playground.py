# backend/scripts/pdf_pymupdf4llm_playground.py

import os
import json
import pymupdf4llm

# Use an environment variable if provided, otherwise default to the local data path.
# Example: PDF_PATH=data/pdf-files-for-upload/Mount_Olympus.pdf python backend/scripts/pdf_pymupdf4llm_playground.py
PDF_PATH = os.environ.get("PDF_PATH", "data/pdf-files-for-upload/Mount_Whitney.pdf")

def main():
    print(f"Using PDF: {PDF_PATH}")

    # --- 1) Whole-document markdown (quick feel for headings / content) ---
    print("\n=== FULL MARKDOWN (first 100 lines) ===")
    md_text = pymupdf4llm.to_markdown(PDF_PATH)
    for i, line in enumerate(md_text.splitlines()):
        print(line)
        if i >= 99:
            break
    print("\n--- [truncated] ---\n")

    # --- 2) Chunked markdown with metadata + tables ---
    print("=== PAGE / CHUNK STRUCTURE SUMMARY ===")
    chunks = pymupdf4llm.to_markdown(
        PDF_PATH,
        page_chunks=True,          # return list[dict]
        table_strategy="lines_strict",  # good default for table detection
    )
    print(f"Total chunks: {len(chunks)}")

    # Show a quick summary of the first few chunks
    for i, ch in enumerate(chunks[:5]):
        text_preview = (ch.get("text") or "").strip().replace("\n", " ")
        if len(text_preview) > 160:
            text_preview = text_preview[:160] + "..."
        tables_count = len(ch.get("tables") or [])
        page = ch.get("metadata", {}).get("page_number")
        print(f"\n--- Chunk {i} ---")
        print(f"Page: {page}")
        print(f"Tables: {tables_count}")
        print(f"Text preview: {text_preview!r}")

    # --- 2a) Full markdown for the first two chunks (approx. first two pages) ---
    if len(chunks) > 0:
        print("\n=== FULL MARKDOWN FOR CHUNK 0 (approx. page 1) ===\n")
        print(chunks[0].get("text", ""))

    if len(chunks) > 1:
        print("\n=== FULL MARKDOWN FOR CHUNK 1 (approx. page 2) ===\n")
        print(chunks[1].get("text", ""))

    # --- 3) Look specifically for sections like 'Geography' / 'Morphology' ---
    print("\n=== CHUNKS CONTAINING 'Geography' OR 'Morphology' ===")
    for i, ch in enumerate(chunks):
        text = ch.get("text") or ""
        if "Geography" in text or "Morphology" in text:
            text_preview = text.strip().replace("\n", " ")
            if len(text_preview) > 260:
                text_preview = text_preview[:260] + "..."
            tables_count = len(ch.get("tables") or [])
            page = ch.get("metadata", {}).get("page_number")
            print(f"\n--- Chunk {i} ---")
            print(f"Page: {page}")
            print(f"Tables: {tables_count}")
            print(f"Text preview: {text_preview!r}")

    # --- 4) Show any chunks that start with a markdown heading (#) ---
    print("\n=== CHUNKS WHOSE TEXT STARTS WITH A HEADING (#) ===")
    for i, ch in enumerate(chunks):
        text = (ch.get("text") or "").lstrip()
        if text.startswith("#"):
            text_preview = text.replace("\n", " ")
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            tables_count = len(ch.get("tables") or [])
            page = ch.get("metadata", {}).get("page_number")
            print(f"\n--- Chunk {i} ---")
            print(f"Page: {page}")
            print(f"Tables: {tables_count}")
            print(f"Text: {text_preview!r}")

    # Optional: dump one chunk with a table as raw JSON to inspect structure
    for ch in chunks:
        if ch.get("tables"):
            print("\n=== SAMPLE CHUNK WITH TABLE ===")
            print(json.dumps(ch, indent=2)[:2000])
            break


if __name__ == "__main__":
    main()