# backend/scripts/pdf_pymupdf4llm_playground.py

import os
import json
import pymupdf4llm

#PDF_PATH = os.environ.get("/app/data/pdf-files-for-upload/", "Mount_Olympus.pdf")  # adjust default as needed
PDF_PATH = "data/pdf-files-for-upload/Mount_Olympus.pdf"
def main():
    print(f"Using PDF: {PDF_PATH}")

    # --- 1) Whole-document markdown (quick feel for headings / content) ---
    print("\n=== FULL MARKDOWN (first 800 chars) ===")
    md_text = pymupdf4llm.to_markdown(PDF_PATH)
    print(md_text[:800])
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

    # Optional: dump one chunk with a table as raw JSON to inspect structure
    for ch in chunks:
        if ch.get("tables"):
            print("\n=== SAMPLE CHUNK WITH TABLE ===")
            print(json.dumps(ch, indent=2)[:2000])
            break


if __name__ == "__main__":
    main()