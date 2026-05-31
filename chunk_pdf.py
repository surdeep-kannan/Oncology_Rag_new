import json
import re
from pathlib import Path

import fitz


PDF_PATH = "adult_cancer_guidelines.pdf"
HEADINGS_PATH = "6.txt"
OUTPUT_PATH = "chunks.jsonl"
MIN_CONTENT_LENGTH = 80
HEADING_MIN_FONT = 19
HEADING_MAX_FONT = 80


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("»", " ")
    text = re.sub(r"[^\w\s-]", " ", text)
    return normalize_whitespace(text)


def clean_heading_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^\d+\s+", "", text)
    return text


def resolve_pdf_path() -> str:
    default_path = Path(PDF_PATH)
    if default_path.exists():
        return str(default_path)

    pdf_files = sorted(Path(".").glob("*.pdf"))
    if len(pdf_files) == 1:
        print(f"Default PDF not found. Using detected file: {pdf_files[0].name}")
        return str(pdf_files[0])

    raise FileNotFoundError(
        f"Could not find '{PDF_PATH}'. Found PDF files: {[pdf.name for pdf in pdf_files]}"
    )


def is_toc_page(block_texts: list[str]) -> bool:
    joined = " ".join(block_texts)
    if "Contents" in joined:
        return True

    toc_entry_count = sum(bool(re.match(r"^\d+\s+\S", text)) for text in block_texts)
    return toc_entry_count >= 4


def extract_blocks(page: fitz.Page) -> list[dict]:
    data = page.get_text("dict")
    blocks = []

    for block_index, block in enumerate(data["blocks"]):
        if block.get("type") != 0:
            continue

        spans = []
        texts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span["text"].strip()
                if span_text:
                    spans.append(span)
                    texts.append(span_text)

        if not texts:
            continue

        text = normalize_whitespace(" ".join(texts))
        max_font = max(span["size"] for span in spans)
        font_names = {span["font"] for span in spans}
        min_y = block["bbox"][1]
        max_y = block["bbox"][3]
        min_x = block["bbox"][0]
        max_x = block["bbox"][2]
        blocks.append(
            {
                "index": block_index,
                "text": text,
                "normalized": normalize_for_match(text),
                "max_font": max_font,
                "font_names": sorted(font_names),
                "min_y": min_y,
                "max_y": max_y,
                "min_x": min_x,
                "max_x": max_x,
            }
        )

    return blocks


def load_document(pdf_path: str) -> list[dict]:
    print("Loading PDF...")
    doc = fitz.open(pdf_path)
    pages = []

    for page_index, page in enumerate(doc):
        blocks = extract_blocks(page)
        pages.append(
            {
                "page_index": page_index,
                "blocks": blocks,
                "is_toc": is_toc_page([block["text"] for block in blocks]),
            }
        )

    print("PDF loaded.")
    return pages


def load_headings() -> list[str]:
    print("Loading headings...")
    titles = []
    with open(HEADINGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Level\s+\d+:\s+(.*)", line.strip())
            if match:
                titles.append(match.group(1).strip())
    print(f"Loaded {len(titles)} headings.")
    return titles


def is_header_or_footer(block: dict) -> bool:
    text = block["text"]
    if block["min_y"] < 55 and "»" in text:
        return True
    if block["max_y"] > 730:
        return True
    if text in {"Ü", "2023"}:
        return True
    if "NCCN Guidelines for Patients" in text and block["max_font"] <= 14:
        return True
    if re.fullmatch(r"\d+", text):
        return True
    return False


def is_heading_candidate(block: dict, page: dict) -> bool:
    if is_header_or_footer(block):
        return False
    if not (HEADING_MIN_FONT <= block["max_font"] <= HEADING_MAX_FONT):
        return False

    text = block["text"]
    normalized = block["normalized"]

    if len(normalized) < 3:
        return False
    if len(text.split()) > 10 and block["max_font"] > 30:
        return False
    if page["is_toc"] and block["max_font"] < 40:
        return False
    if re.fullmatch(r"[“”\"]+", text):
        return False
    if text == "share with us.":
        return False
    return True


def split_breadcrumb(text: str) -> list[str]:
    parts = [normalize_whitespace(part) for part in text.split("»")]
    return [part for part in parts if part]


def find_breadcrumb_for_heading(page: dict, heading_block: dict) -> list[str]:
    matches = []
    heading_norm = normalize_for_match(clean_heading_text(heading_block["text"]))

    for block in page["blocks"]:
        if block["min_y"] >= 60:
            continue
        if "»" not in block["text"]:
            continue
        if block["index"] > heading_block["index"]:
            continue

        breadcrumb_parts = split_breadcrumb(block["text"])
        breadcrumb_norms = [normalize_for_match(part) for part in breadcrumb_parts]
        if not breadcrumb_norms:
            continue

        last_norm = breadcrumb_norms[-1]
        score = 0
        if heading_norm == last_norm:
            score = 3
        elif heading_norm in last_norm or last_norm in heading_norm:
            score = 2
        elif any(heading_norm == crumb or heading_norm in crumb or crumb in heading_norm for crumb in breadcrumb_norms):
            score = 1

        if score:
            matches.append((score, block["index"], breadcrumb_parts))

    if matches:
        matches.sort(key=lambda item: (item[0], item[1]))
        return matches[-1][2]
    return []


def detect_anchors(pages: list[dict]) -> list[dict]:
    print("Detecting section anchors...")
    anchors = []

    for page in pages:
        for block in page["blocks"]:
            if not is_heading_candidate(block, page):
                continue

            heading_text = clean_heading_text(block["text"])
            breadcrumb_parts = find_breadcrumb_for_heading(page, block)

            if page["is_toc"] and block["max_font"] < 40:
                continue

            if breadcrumb_parts:
                heading = breadcrumb_parts[-1]
            else:
                heading = heading_text

            anchors.append(
                {
                    "page_index": page["page_index"],
                    "block_index": block["index"],
                    "heading": heading,
                    "heading_text": heading_text,
                    "breadcrumb_parts": breadcrumb_parts,
                    "min_y": block["min_y"],
                }
            )

    print(f"Detected {len(anchors)} anchors.")
    return anchors


def match_hierarchy(anchor: dict, heading_titles: list[str]) -> dict:
    breadcrumb = anchor["breadcrumb_parts"][:]
    if not breadcrumb:
        breadcrumb = [anchor["heading"]]

    levels = {"level1": None, "level2": None, "level3": None, "level4": None}
    for index, part in enumerate(breadcrumb[:4], start=1):
        levels[f"level{index}"] = part

    if not any(levels.values()):
        anchor_norm = normalize_for_match(anchor["heading"])
        for title in heading_titles:
            if normalize_for_match(title) == anchor_norm:
                levels["level1"] = title
                break

    return levels


def is_bibliography_or_reference_block(block: dict) -> bool:
    """Detect if a block consists strictly of citations, references, or bibliography lists."""
    text = block["text"]
    words = text.split()
    if len(words) < 3:
        return False

    # 1. High density of standard clinical publication years: e.g. (2008), (1999)
    years = len(re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text))
    
    # 2. Medical journals, guidelines, and database abbreviations
    journals = len(re.findall(r'\b(J Clin Oncol|N Engl J Med|Lancet|JAMA|Oncol|Med|Surg|N Engl|bmj|guideline|guidelines|ASCO|NCCN|WHO)\b', text, re.IGNORECASE))
    
    # 3. Reference item numbers: e.g., "1. Barry MJ", "[1] Barry MJ"
    list_items = len(re.findall(r'^\b\d+[\.\)]\s+', text)) + len(re.findall(r'^\[\d+\]\s+', text))
    
    # 4. Standard citation markers: e.g. "et al.", "pp. ", "vol. "
    citation_markers = len(re.findall(r'\b(et al\.|pp\.|vol\.|journal|clinical trial|recommendation|consensus)\b', text, re.IGNORECASE))

    # If the block has high concentration of years AND journals/citation markers, or lists reference items:
    if (years >= 2 and journals >= 1) or (years >= 1 and citation_markers >= 2) or list_items >= 2:
        return True
        
    return False


def should_keep_block(block: dict) -> bool:
    if is_header_or_footer(block):
        return False
    text = block["text"]
    if not text:
        return False
    if text == "share with us.":
        return False
    if is_visual_noise_block(block):
        return False
    if is_bibliography_or_reference_block(block):
        return False
    return True


def is_visual_noise_block(block: dict) -> bool:
    text = block["text"]
    words = text.split()
    font_names = " ".join(block.get("font_names", []))

    if "AmericanTypewriter" in font_names:
        return True
    if "Wingdings" in font_names and len(words) <= 12:
        return True
    if block["max_font"] >= 13.5 and len(words) <= 20:
        return True
    if block["max_font"] >= 15 and block["min_x"] > 300:
        return True
    if text.count("http") or text.count(".org") + text.count(".gov") >= 2:
        return True
    if len(words) <= 8 and block["min_x"] > 300:
        return True
    return False


def is_lead_in_block(block: dict) -> bool:
    if not should_keep_block(block):
        return False
    return 12.5 <= block["max_font"] < HEADING_MIN_FONT


def is_toc_like_content(text: str) -> bool:
    snippet = text[:220]
    return len(re.findall(r"\b\d+\s+[A-Za-z]", snippet)) >= 3


def extract_content_between_anchors(pages: list[dict], anchors: list[dict], idx: int) -> str:
    current = anchors[idx]
    next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
    pieces = []

    previous_same_page = None
    for j in range(idx - 1, -1, -1):
        if anchors[j]["page_index"] == current["page_index"]:
            previous_same_page = anchors[j]
            break
        if anchors[j]["page_index"] < current["page_index"]:
            break

    previous_anchor = anchors[idx - 1] if idx > 0 else None

    for page_index in range(current["page_index"], len(pages)):
        page = pages[page_index]
        for block in page["blocks"]:
            if not should_keep_block(block):
                continue

            if page_index == current["page_index"]:
                if previous_same_page and block["index"] <= previous_same_page["block_index"]:
                    continue
                if not previous_same_page and block["min_y"] < 60:
                    continue
                if block["index"] < current["block_index"]:
                    previous_page_is_intro = previous_anchor is None or pages[previous_anchor["page_index"]]["is_toc"]
                    if not previous_page_is_intro or not is_lead_in_block(block):
                        continue
                if block["index"] == current["block_index"]:
                    continue

            if next_anchor and page_index == next_anchor["page_index"] and block["index"] >= next_anchor["block_index"]:
                return normalize_whitespace(" ".join(pieces))

            pieces.append(block["text"])

        if next_anchor and page_index >= next_anchor["page_index"]:
            break

    return normalize_whitespace(" ".join(pieces))


def build_chunks(pages: list[dict], anchors: list[dict], heading_titles: list[str]) -> list[dict]:
    print("Creating chunks...")
    chunks = []

    for idx, anchor in enumerate(anchors):
        if anchor["page_index"] == 0:
            continue
        content = extract_content_between_anchors(pages, anchors, idx)
        if len(content) < MIN_CONTENT_LENGTH:
            continue
        if is_toc_like_content(content):
            continue

        levels = match_hierarchy(anchor, heading_titles)
        chunks.append(
            {
                "level1": levels["level1"],
                "level2": levels["level2"],
                "level3": levels["level3"],
                "level4": levels["level4"],
                "heading": anchor["heading"],
                "content": content,
            }
        )

    print(f"Created {len(chunks)} chunks.")
    return chunks


def save_chunks(chunks: list[dict]) -> None:
    print("Saving chunks...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_PATH}")


def main() -> None:
    pdf_path = resolve_pdf_path()
    pages = load_document(pdf_path)
    heading_titles = load_headings()
    anchors = detect_anchors(pages)
    chunks = build_chunks(pages, anchors, heading_titles)
    save_chunks(chunks)
    print("\nDONE")


if __name__ == "__main__":
    main()
