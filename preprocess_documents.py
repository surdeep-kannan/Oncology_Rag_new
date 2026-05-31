"""Structure-aware preprocessing for medical RAG documents.

This script converts noisy raw documents into hierarchy-aware cleaned JSONL
records that can feed semantic chunking. It focuses on deterministic parsing
and cleanup rather than retrieval or LLM-based processing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import fitz


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "cleaned_documents.jsonl"
DEFAULT_INPUT_CANDIDATES = (
    "adult cancer guidelinespdf.pdf",
    "6.txt",
    "sai.txt",
    "chunks.jsonl",
)
TEXT_KEYS = ("cleaned_text", "content", "text", "body", "value")
MAX_HEADING_WORDS = 12
MIN_HEADING_FONT_DELTA = 1.5
SUSPICIOUS_SENTENCE_STARTS = (
    "If",
    "When",
    "After",
    "Before",
    "Some",
    "These",
    "Those",
    "This",
    "That",
    "You",
    "Your",
    "They",
    "It",
    "Talk",
    "Ask",
    "More",
    "Cancer",
    "Treatment",
    "Supportive",
    "Fertility",
)


@dataclass
class PreprocessStats:
    removed_headers_footers: int = 0
    toc_fragments_removed: int = 0
    malformed_joins_repaired: int = 0
    rejected_false_headings: int = 0
    suspicious_merges_remaining: int = 0
    duplicate_whitespace_removed: int = 0


@dataclass(frozen=True)
class PdfBlock:
    text: str
    page_index: int
    block_index: int
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    max_font: float
    font_names: tuple[str, ...]


@dataclass(frozen=True)
class SectionNode:
    source_file: str
    level1: str | None
    level2: str | None
    level3: str | None
    section_type: str
    cleaned_text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PDF/TXT/MD/JSON/JSONL documents into cleaned hierarchy-aware JSONL."
    )
    parser.add_argument("input", nargs="?", default=None, help="Input file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT.name}",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "pdf", "txt", "md", "json", "jsonl"),
        default="auto",
        help="Force an input format. Default: auto",
    )
    return parser.parse_args()


def resolve_input_path(user_input: str | None) -> Path:
    if user_input:
        path = Path(user_input)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        path = BASE_DIR / candidate
        if path.exists():
            return path

    tried = ", ".join(DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find a default input file. Tried: {tried}")


def detect_input_format(path: Path, forced: str) -> str:
    if forced != "auto":
        return forced
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix == ".md":
        return "md"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"Unsupported input format for file: {path}")


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("»", " ")
    text = re.sub(r"[^\w\s/-]", " ", text)
    return normalize_spaces(text)


def count_words(text: str) -> int:
    return len(text.split())


def looks_like_url(text: str) -> bool:
    return "http" in text.lower() or ".org" in text.lower() or ".edu" in text.lower()


def looks_like_navigation(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "available online",
            "connect with us",
            "find an nccn cancer",
            "nccn.org",
            "patientguidelines",
            "please take a moment",
        )
    )


def looks_like_author_or_institution(text: str) -> bool:
    if looks_like_url(text):
        return True
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "university",
            "cancer center",
            "hospital",
            "institute",
            "foundation",
            "medical center",
            "senior",
            "director",
            "writer",
            "specialist",
        )
    ):
        return True
    if len(re.findall(r"\b[A-Z][a-z]+\b", text)) >= 3 and "," in text:
        return True
    return False


def is_symbol_noise(text: str) -> bool:
    stripped = text.strip()
    if stripped in {"Ü", "®", "*", "-", "•"}:
        return True
    return bool(re.fullmatch(r"[®Ü•*=\-_/|]+", stripped))


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n", normalized)
    return [part.strip() for part in parts if part.strip()]


def is_bullet_line(text: str) -> bool:
    return bool(re.match(r"^([-*•]|h |\d+[.)]|[A-Za-z][.)])\s+", text.strip()))


def is_toc_line(text: str) -> bool:
    stripped = normalize_spaces(text)
    return bool(
        re.match(r"^\d+\s+[A-Z].*", stripped)
        or re.match(r"^[A-Z][A-Za-z'()/,& -]+\s+\d{1,2}(?:[-–]\d{1,2})?$", stripped)
    )


def extract_pdf_blocks(doc: fitz.Document) -> list[PdfBlock]:
    blocks: list[PdfBlock] = []
    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block_index, raw_block in enumerate(page_dict.get("blocks", [])):
            if raw_block.get("type") != 0:
                continue

            spans: list[dict[str, Any]] = []
            texts: list[str] = []
            for line in raw_block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        texts.append(text)
                        spans.append(span)

            if not texts:
                continue

            text = normalize_spaces(" ".join(texts))
            if not text:
                continue

            blocks.append(
                PdfBlock(
                    text=text,
                    page_index=page_index,
                    block_index=block_index,
                    min_x=float(raw_block["bbox"][0]),
                    min_y=float(raw_block["bbox"][1]),
                    max_x=float(raw_block["bbox"][2]),
                    max_y=float(raw_block["bbox"][3]),
                    max_font=max(float(span["size"]) for span in spans),
                    font_names=tuple(sorted({str(span["font"]) for span in spans})),
                )
            )
    return blocks


def detect_repeated_margin_texts(doc: fitz.Document, blocks: list[PdfBlock]) -> set[str]:
    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()

    for block in blocks:
        page_height = doc[block.page_index].rect.height
        normalized = normalize_for_match(block.text)
        if not normalized:
            continue
        if block.min_y < 55:
            top_counts[normalized] += 1
        if block.max_y > page_height - 35:
            bottom_counts[normalized] += 1

    repeated = {
        text
        for text, count in {**top_counts, **bottom_counts}.items()
        if count >= 2 and len(text.split()) <= 10
    }
    return repeated


def is_toc_page(page_blocks: list[PdfBlock]) -> bool:
    texts = [block.text for block in page_blocks]
    joined = " ".join(texts).lower()
    toc_lines = sum(1 for text in texts if is_toc_line(text))
    return "contents" in joined or toc_lines >= 5


def body_font_size(blocks: list[PdfBlock]) -> float:
    candidates = [block.max_font for block in blocks if len(block.text.split()) > 8]
    return median(candidates) if candidates else 11.0


def heading_score(block: PdfBlock, body_font: float) -> int:
    text = block.text.strip()
    word_count = len(text.split())
    score = 0

    if word_count == 0 or word_count > MAX_HEADING_WORDS:
        return -10
    if text.endswith((".", "!", "?")):
        score -= 2
    if looks_like_author_or_institution(text) or looks_like_navigation(text):
        score -= 4
    if is_bullet_line(text) or looks_like_url(text):
        score -= 4
    if re.search(r"\d{4}", text) and word_count > 4:
        score -= 2

    if block.max_font >= body_font + MIN_HEADING_FONT_DELTA:
        score += 4
    if re.match(r"^\d+(\.\d+)*\s+", text):
        score += 3
    if text == text.upper() and word_count <= 8:
        score += 2
    if sum(word[:1].isupper() for word in text.split()) >= max(2, word_count - 1):
        score += 2

    return score


def classify_heading_level(text: str, breadcrumb: list[str]) -> tuple[str | None, str | None, str | None]:
    cleaned = normalize_spaces(text)
    if breadcrumb:
        parts = breadcrumb[:3]
        if parts[-1] != cleaned:
            parts.append(cleaned)
        parts = parts[:3]
        parts += [None] * (3 - len(parts))
        return parts[0], parts[1], parts[2]

    if re.match(r"^\d+\s+", cleaned):
        return cleaned, None, None
    return cleaned, None, None


def split_breadcrumb(text: str) -> list[str]:
    return [normalize_spaces(part) for part in text.split("»") if normalize_spaces(part)]


def should_filter_pdf_block(
    block: PdfBlock,
    repeated_margins: set[str],
    body_font: float,
    page_height: float,
    toc_page: bool,
    stats: PreprocessStats,
) -> bool:
    normalized = normalize_for_match(block.text)
    if normalized in repeated_margins:
        stats.removed_headers_footers += 1
        return True
    if is_symbol_noise(block.text):
        stats.removed_headers_footers += 1
        return True
    if re.fullmatch(r"\d+", block.text.strip()):
        stats.removed_headers_footers += 1
        return True
    if looks_like_navigation(block.text):
        stats.removed_headers_footers += 1
        return True
    if block.max_y > page_height - 28:
        stats.removed_headers_footers += 1
        return True
    if toc_page:
        stats.toc_fragments_removed += 1
        return True
    if is_toc_line(block.text):
        stats.toc_fragments_removed += 1
        return True
    return False


def merge_wrapped_lines(lines: list[str], stats: PreprocessStats) -> list[str]:
    if not lines:
        return []

    merged: list[str] = [lines[0]]
    for line in lines[1:]:
        previous = merged[-1]
        if is_bullet_line(line) or line.startswith("#"):
            merged.append(line)
            continue
        if re.search(r"[a-z,;:]$", previous) and re.match(r"^[a-z(]", line):
            merged[-1] = f"{previous} {line}"
            stats.malformed_joins_repaired += 1
            continue
        if re.search(r"[A-Za-z]$", previous) and re.match(r"^[A-Z][a-z]", line):
            merged.append(line)
            continue
        merged[-1] = f"{previous} {line}"
        stats.malformed_joins_repaired += 1
    return merged


def repair_mid_sentence_caps(paragraph: str, stats: PreprocessStats) -> str:
    pattern = r"(?<=[a-z])\s+(?=(%s)\b)" % "|".join(re.escape(token) for token in SUSPICIOUS_SENTENCE_STARTS)
    if re.search(pattern, paragraph):
        paragraph = re.sub(pattern, ". ", paragraph)
        stats.malformed_joins_repaired += 1
    return paragraph


def clean_paragraph(paragraph: str, stats: PreprocessStats) -> str:
    original = paragraph
    before_spaces = len(re.findall(r"[ \t]{2,}", paragraph))
    if before_spaces:
        stats.duplicate_whitespace_removed += before_spaces
    paragraph = normalize_spaces(paragraph)
    paragraph = paragraph.replace("\u00ad", "")
    paragraph = paragraph.replace(" .", ".")
    paragraph = repair_mid_sentence_caps(paragraph, stats)
    if paragraph != original.strip():
        stats.malformed_joins_repaired += 0
    return paragraph


def finalize_section(
    sections: list[SectionNode],
    source_file: str,
    breadcrumb: list[str],
    heading_text: str | None,
    section_type: str,
    lines: list[str],
    stats: PreprocessStats,
) -> None:
    cleaned_lines = merge_wrapped_lines([line for line in lines if line.strip()], stats)
    cleaned_paragraphs = [clean_paragraph(line, stats) for line in cleaned_lines if clean_paragraph(line, stats)]
    if not cleaned_paragraphs:
        return

    level1, level2, level3 = classify_heading_level(heading_text or (breadcrumb[-1] if breadcrumb else source_file), breadcrumb)
    node = SectionNode(
        source_file=source_file,
        level1=level1,
        level2=level2,
        level3=level3,
        section_type=section_type,
        cleaned_text="\n\n".join(cleaned_paragraphs),
        metadata={
            "heading": heading_text or level3 or level2 or level1,
            "paragraph_count": len(cleaned_paragraphs),
            "word_count": count_words(" ".join(cleaned_paragraphs)),
        },
    )
    stats.suspicious_merges_remaining += count_suspicious_merges(node.cleaned_text)
    sections.append(node)


def count_suspicious_merges(text: str) -> int:
    count = 0
    for paragraph in text.split("\n\n"):
        if re.search(r"[a-z]\s+[A-Z][a-z]+", paragraph) and not re.search(r"[.!?]\s+[A-Z]", paragraph):
            count += 1
        if len(paragraph.split()) <= 6 and re.search(r"\b(and|but|because|or|so|then)\b", paragraph, re.IGNORECASE):
            count += 1
    return count


def load_pdf_records(path: Path, stats: PreprocessStats) -> list[dict[str, Any]]:
    doc = fitz.open(path)
    blocks = extract_pdf_blocks(doc)
    if not blocks:
        return []

    body_font = body_font_size(blocks)
    repeated_margins = detect_repeated_margin_texts(doc, blocks)
    pages: dict[int, list[PdfBlock]] = defaultdict(list)
    for block in blocks:
        pages[block.page_index].append(block)

    sections: list[SectionNode] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    current_section_type = "section"
    current_breadcrumb: list[str] = []

    for page_index in sorted(pages):
        page_blocks = sorted(pages[page_index], key=lambda block: (block.min_y, block.min_x))
        toc_page = is_toc_page(page_blocks)
        page_height = doc[page_index].rect.height

        for block in page_blocks:
            if should_filter_pdf_block(block, repeated_margins, body_font, page_height, toc_page, stats):
                continue

            if "»" in block.text and block.min_y < 80:
                current_breadcrumb = split_breadcrumb(block.text)
                continue

            score = heading_score(block, body_font)
            if score >= 4:
                finalize_section(
                    sections,
                    path.name,
                    current_breadcrumb,
                    current_heading,
                    current_section_type,
                    current_lines,
                    stats,
                )
                current_heading = block.text.strip()
                current_lines = []
                current_section_type = "subsection" if current_breadcrumb else "section"
                continue

            if score <= -2 and looks_like_author_or_institution(block.text):
                stats.rejected_false_headings += 1

            current_lines.append(block.text)

    finalize_section(
        sections,
        path.name,
        current_breadcrumb,
        current_heading,
        current_section_type,
        current_lines,
        stats,
    )
    return [section_to_record(section) for section in sections if section.cleaned_text.strip()]


def section_to_record(section: SectionNode) -> dict[str, Any]:
    return {
        "level1": section.level1,
        "level2": section.level2,
        "level3": section.level3,
        "section_type": section.section_type,
        "cleaned_text": section.cleaned_text,
        "metadata": {
            "source_file": section.source_file,
            **section.metadata,
        },
    }


def text_to_hierarchy_records(path: Path, text: str, stats: PreprocessStats) -> list[dict[str, Any]]:
    raw_paragraphs = split_paragraphs(text)
    lines = [paragraph for paragraph in raw_paragraphs if paragraph.strip()]

    sections: list[SectionNode] = []
    current_heading: str | None = path.stem
    current_lines: list[str] = []
    current_hierarchy: list[str] = [path.stem]

    for line in lines:
        normalized = normalize_spaces(line)
        if not normalized:
            continue
        if is_symbol_noise(normalized) or looks_like_navigation(normalized):
            stats.removed_headers_footers += 1
            continue
        if is_toc_line(normalized):
            stats.toc_fragments_removed += 1
            continue

        if looks_like_heading_line(normalized):
            finalize_section(
                sections,
                path.name,
                current_hierarchy,
                current_heading,
                "section",
                current_lines,
                stats,
            )
            current_heading = normalized.lstrip("# ").strip()
            current_hierarchy = [current_heading]
            current_lines = []
            continue

        current_lines.append(normalized)

    finalize_section(
        sections,
        path.name,
        current_hierarchy,
        current_heading,
        "section",
        current_lines,
        stats,
    )
    return [section_to_record(section) for section in sections if section.cleaned_text.strip()]


def looks_like_heading_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if is_bullet_line(stripped) or looks_like_url(stripped):
        return False
    if looks_like_author_or_institution(stripped):
        return False
    if stripped.endswith((".", "!", "?")):
        return False
    words = stripped.split()
    if len(words) > MAX_HEADING_WORDS:
        return False
    if re.match(r"^\d+(\.\d+)*\s+", stripped):
        return True
    return sum(word[:1].isupper() for word in words) >= max(2, len(words) - 1)


def extract_text_field(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def extract_levels(item: dict[str, Any], source_file: str) -> tuple[str | None, str | None, str | None, str]:
    level1 = as_optional_string(item.get("level1"))
    level2 = as_optional_string(item.get("level2"))
    level3 = as_optional_string(item.get("level3")) or as_optional_string(item.get("heading"))
    heading = as_optional_string(item.get("heading")) or level3 or level2 or level1 or source_file
    return level1, level2, level3 or heading, heading


def as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_json_records(path: Path, stats: PreprocessStats) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("sections") if isinstance(data, dict) and isinstance(data.get("sections"), list) else data
    if not isinstance(items, list):
        items = [items]

    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_text = extract_text_field(item)
        if not raw_text:
            continue
        paragraphs = [clean_paragraph(paragraph, stats) for paragraph in split_paragraphs(raw_text) if paragraph.strip()]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        if not paragraphs:
            continue
        level1, level2, level3, heading = extract_levels(item, path.name)
        stats.suspicious_merges_remaining += count_suspicious_merges("\n\n".join(paragraphs))
        records.append(
            {
                "level1": level1,
                "level2": level2,
                "level3": level3,
                "section_type": "section",
                "cleaned_text": "\n\n".join(paragraphs),
                "metadata": {
                    "source_file": path.name,
                    "heading": heading,
                    "paragraph_count": len(paragraphs),
                    "word_count": count_words(" ".join(paragraphs)),
                },
            }
        )
    return records


def load_jsonl_records(path: Path, stats: PreprocessStats) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            raw_text = extract_text_field(item)
            if not raw_text:
                continue
            paragraphs = [clean_paragraph(paragraph, stats) for paragraph in split_paragraphs(raw_text) if paragraph.strip()]
            paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            if not paragraphs:
                continue
            level1, level2, level3, heading = extract_levels(item, path.name)
            stats.suspicious_merges_remaining += count_suspicious_merges("\n\n".join(paragraphs))
            records.append(
                {
                    "level1": level1,
                    "level2": level2,
                    "level3": level3,
                    "section_type": "section",
                    "cleaned_text": "\n\n".join(paragraphs),
                    "metadata": {
                        "source_file": path.name,
                        "heading": heading,
                        "paragraph_count": len(paragraphs),
                        "word_count": count_words(" ".join(paragraphs)),
                    },
                }
            )
    return records


def load_records(path: Path, input_format: str, stats: PreprocessStats) -> list[dict[str, Any]]:
    if input_format == "pdf":
        return load_pdf_records(path, stats)
    if input_format in {"txt", "md"}:
        return text_to_hierarchy_records(path, path.read_text(encoding="utf-8"), stats)
    if input_format == "json":
        return load_json_records(path, stats)
    if input_format == "jsonl":
        return load_jsonl_records(path, stats)
    raise ValueError(f"Unsupported input format: {input_format}")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_report(
    input_path: Path,
    output_path: Path,
    input_format: str,
    records: list[dict[str, Any]],
    stats: PreprocessStats,
) -> None:
    print(f"Input file: {input_path}")
    print(f"Input format: {input_format}")
    print(f"Output file: {output_path}")
    print(f"Records written: {len(records)}")
    print("Validation report:")
    print(f"  Removed headers/footers: {stats.removed_headers_footers}")
    print(f"  TOC fragments removed: {stats.toc_fragments_removed}")
    print(f"  Malformed joins repaired: {stats.malformed_joins_repaired}")
    print(f"  Rejected false headings: {stats.rejected_false_headings}")
    print(f"  Suspicious merges remaining: {stats.suspicious_merges_remaining}")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, str]:
    input_path = resolve_input_path(args.input)
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output
    input_format = detect_input_format(input_path, args.input_format)
    return input_path, output_path, input_format


def main() -> None:
    args = parse_args()
    input_path, output_path, input_format = validate_args(args)
    stats = PreprocessStats()
    records = load_records(input_path, input_format, stats)
    write_jsonl(records, output_path)
    print_report(input_path, output_path, input_format, records, stats)


if __name__ == "__main__":
    main()
