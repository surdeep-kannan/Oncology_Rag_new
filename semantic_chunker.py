"""Deterministic hierarchical semantic chunker for medical RAG pipelines.

Supported inputs:
- PDF
- TXT
- JSON
- JSONL

The chunker preserves section hierarchy, respects sentence boundaries, keeps
paragraph structure when possible, and emits metadata-rich JSONL output that is
well suited for dense retrieval, BM25, hybrid search, rerankers, and QA.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import fitz
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "semantic_chunks.jsonl"
DEFAULT_MIN_WORDS = 80
DEFAULT_TARGET_WORDS = 160
DEFAULT_MAX_WORDS = 240
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_INPUT_CANDIDATES = (
    "chunks.jsonl",
    "6.jsonl",
    "adult cancer guidelinespdf.pdf",
    "6.txt",
)
SENTENCE_ENDINGS = (".", "!", "?", '"', "'", "”", "’", ")", "]")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "not",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
    "you",
    "your",
}
TRANSITION_STARTERS = {
    "and",
    "but",
    "because",
    "so",
    "then",
    "therefore",
    "however",
    "also",
    "instead",
    "meanwhile",
    "moreover",
    "otherwise",
    "plus",
    "yet",
    "thus",
}
CONTEXT_LIGHT_PRONOUNS = {
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "them",
    "their",
    "its",
}


@dataclass(frozen=True)
class SectionRecord:
    """One hierarchy-bounded section to chunk."""

    source_file: str
    level1: str | None
    level2: str | None
    level3: str | None
    level4: str | None
    heading: str | None
    paragraphs: list[str]


@dataclass(frozen=True)
class SentenceUnit:
    """A sentence paired with section-local structure metadata."""

    text: str
    paragraph_index: int
    sentence_index: int

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    @property
    def terms(self) -> set[str]:
        return content_terms(self.text)


@dataclass(frozen=True)
class ChunkRange:
    """Inclusive-exclusive sentence range for one chunk."""

    start: int
    end: int


@dataclass(frozen=True)
class ChunkValidation:
    """Validation summary for emitted chunks."""

    empty_chunks: int
    ending_without_punctuation: int
    starts_lowercase_unexpectedly: int
    starts_with_transition_word: int
    oversized_chunks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create hierarchical sentence-aware semantic chunks for a medical RAG pipeline."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input file path. Supports PDF, TXT, JSON, JSONL. Defaults to a detected project file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT.name}",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "pdf", "txt", "json", "jsonl"),
        default="auto",
        help="Force an input format. Default: auto",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum target chunk size. Default: {DEFAULT_MIN_WORDS}",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=DEFAULT_TARGET_WORDS,
        help=f"Preferred chunk size before considering a break. Default: {DEFAULT_TARGET_WORDS}",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help=f"Maximum preferred chunk size without splitting a sentence. Default: {DEFAULT_MAX_WORDS}",
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=DEFAULT_OVERLAP_SENTENCES,
        help=f"Trailing sentence overlap between adjacent chunks. Default: {DEFAULT_OVERLAP_SENTENCES}",
    )
    return parser.parse_args()


def resolve_input_path(user_input: str | None) -> Path:
    if user_input:
        candidate = Path(user_input)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        return candidate

    for candidate_name in DEFAULT_INPUT_CANDIDATES:
        candidate = BASE_DIR / candidate_name
        if candidate.exists():
            return candidate

    tried = ", ".join(DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find a default input file. Tried: {tried}")


def detect_input_format(path: Path, forced_format: str) -> str:
    if forced_format != "auto":
        return forced_format

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"

    raise ValueError(f"Unsupported input format for file: {path}")


def count_words(text: str) -> int:
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_paragraph(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def split_text_into_paragraphs(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    raw_paragraphs = re.split(r"\n\s*\n", cleaned)
    paragraphs = [clean_paragraph(paragraph) for paragraph in raw_paragraphs if clean_paragraph(paragraph)]
    if paragraphs:
        return paragraphs

    return [clean_paragraph(cleaned)] if clean_paragraph(cleaned) else []


def content_terms(text: str) -> set[str]:
    terms = set()
    for token in TOKEN_PATTERN.findall(text.lower()):
        normalized = normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            terms.add(normalized)
    return terms


def normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def build_sentence_tokenizer() -> PunktSentenceTokenizer:
    punkt_params = PunktParameters()
    punkt_params.abbrev_types = {
        "dr",
        "mr",
        "mrs",
        "ms",
        "vs",
        "fig",
        "prof",
        "inc",
        "jr",
        "sr",
        "st",
        "no",
        "appt",
        "hcp",
        "mg",
        "ml",
        "oz",
        "dept",
        "etc",
    }
    return PunktSentenceTokenizer(punkt_params)


def paragraph_to_sentences(paragraph: str, tokenizer: PunktSentenceTokenizer) -> list[str]:
    flat = normalize_whitespace(paragraph.replace("\n", " "))
    if not flat:
        return []
    sentences = [sentence.strip() for sentence in tokenizer.tokenize(flat) if sentence.strip()]
    return sentences or [flat]


def section_to_sentences(section: SectionRecord, tokenizer: PunktSentenceTokenizer) -> list[SentenceUnit]:
    sentences: list[SentenceUnit] = []
    sentence_index = 0
    for paragraph_index, paragraph in enumerate(section.paragraphs):
        for sentence in paragraph_to_sentences(paragraph, tokenizer):
            sentences.append(
                SentenceUnit(
                    text=sentence,
                    paragraph_index=paragraph_index,
                    sentence_index=sentence_index,
                )
            )
            sentence_index += 1
    return repair_sentence_units(sentences)


def is_orphan_sentence_fragment(text: str) -> bool:
    stripped = text.strip()
    if count_words(stripped) > 5:
        return False
    if not stripped:
        return True
    first_alpha = next((character for character in stripped if character.isalpha()), "")
    if not first_alpha or not first_alpha.islower():
        return False
    return bool(re.fullmatch(r"[a-z][a-z,\s-]*[.,;:!?]?", stripped))


def repair_sentence_units(sentences: list[SentenceUnit]) -> list[SentenceUnit]:
    """Drop obviously orphaned source fragments without altering sentence boundaries."""
    repaired: list[SentenceUnit] = []
    next_index = 0
    for unit in sentences:
        if is_orphan_sentence_fragment(unit.text):
            continue
        repaired.append(
            SentenceUnit(
                text=unit.text,
                paragraph_index=unit.paragraph_index,
                sentence_index=next_index,
            )
        )
        next_index += 1
    return repaired


def render_chunk(sentences: Iterable[SentenceUnit]) -> str:
    units = list(sentences)
    if not units:
        return ""

    parts: list[str] = []
    previous_paragraph: int | None = None
    for unit in units:
        if previous_paragraph is None:
            parts.append(unit.text)
        elif unit.paragraph_index != previous_paragraph:
            parts.append("\n\n" + unit.text)
        else:
            parts.append(" " + unit.text)
        previous_paragraph = unit.paragraph_index
    return "".join(parts).strip()


def first_word(text: str) -> str:
    match = re.search(r"[A-Za-z']+", text)
    return match.group(0).lower() if match else ""


def starts_lowercase_unexpectedly(text: str) -> bool:
    stripped = text.lstrip('("“')
    if not stripped:
        return False
    first_char = stripped[0]
    return first_char.isalpha() and first_char.islower()


def is_transition_start(text: str) -> bool:
    return first_word(text) in TRANSITION_STARTERS


def is_context_light_pronoun_start(text: str) -> bool:
    return first_word(text) in CONTEXT_LIGHT_PRONOUNS


def is_awkward_chunk_start(unit: SentenceUnit) -> bool:
    text = unit.text.strip()
    if not text:
        return True
    return (
        is_transition_start(text)
        or starts_lowercase_unexpectedly(text)
        or (is_context_light_pronoun_start(text) and unit.word_count <= 12)
    )


def is_heading_like(paragraph: str) -> bool:
    text = normalize_whitespace(paragraph)
    if not text:
        return False
    if text.startswith("#"):
        return True
    if len(text.split()) > 12:
        return False
    if text.endswith((".", "!", "?")):
        return False
    letters = [character for character in text if character.isalpha()]
    uppercase_ratio = 0.0
    if letters:
        uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    title_words = sum(word[:1].isupper() for word in text.split() if word[:1].isalpha())
    return uppercase_ratio >= 0.6 or title_words >= max(2, len(text.split()) - 1)


def make_section(
    source_file: str,
    content: str | None,
    *,
    level1: str | None = None,
    level2: str | None = None,
    level3: str | None = None,
    level4: str | None = None,
    heading: str | None = None,
    paragraphs: list[str] | None = None,
) -> SectionRecord | None:
    resolved_paragraphs = paragraphs[:] if paragraphs is not None else split_text_into_paragraphs(content or "")
    resolved_paragraphs = [paragraph for paragraph in resolved_paragraphs if paragraph.strip()]
    if not resolved_paragraphs:
        return None

    final_heading = heading or level4 or level3 or level2 or level1 or Path(source_file).stem
    return SectionRecord(
        source_file=source_file,
        level1=level1,
        level2=level2,
        level3=level3,
        level4=level4,
        heading=final_heading,
        paragraphs=resolved_paragraphs,
    )


def load_txt_sections(path: Path) -> list[SectionRecord]:
    text = path.read_text(encoding="utf-8")
    paragraphs = split_text_into_paragraphs(text)
    if not paragraphs:
        return []

    sections: list[SectionRecord] = []
    current_heading: str | None = None
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if is_heading_like(paragraph):
            if current_paragraphs:
                section = make_section(
                    path.name,
                    None,
                    level1=current_heading or path.stem,
                    heading=current_heading or path.stem,
                    paragraphs=current_paragraphs,
                )
                if section:
                    sections.append(section)
                current_paragraphs = []
            current_heading = paragraph.lstrip("# ").strip()
            continue
        current_paragraphs.append(paragraph)

    if current_paragraphs:
        section = make_section(
            path.name,
            None,
            level1=current_heading or path.stem,
            heading=current_heading or path.stem,
            paragraphs=current_paragraphs,
        )
        if section:
            sections.append(section)

    return sections


def extract_text_value(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("cleaned_text", "content", "text", "body", "value"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def extract_paragraphs_value(data: Any) -> list[str] | None:
    if not isinstance(data, dict):
        return None
    paragraphs = data.get("paragraphs")
    if isinstance(paragraphs, list):
        return [clean_paragraph(str(paragraph)) for paragraph in paragraphs if str(paragraph).strip()]
    return None


def section_from_mapping(data: dict[str, Any], source_file: str) -> SectionRecord | None:
    paragraphs = extract_paragraphs_value(data)
    content = extract_text_value(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return make_section(
        source_file,
        content,
        level1=as_optional_string(data.get("level1")),
        level2=as_optional_string(data.get("level2")),
        level3=as_optional_string(data.get("level3")),
        level4=as_optional_string(data.get("level4")),
        heading=(
            as_optional_string(data.get("heading"))
            or as_optional_string(data.get("title"))
            or as_optional_string(metadata.get("heading"))
        ),
        paragraphs=paragraphs,
    )


def as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_json_sections(path: Path) -> list[SectionRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sections_from_json_like(data, path.name)


def load_jsonl_sections(path: Path) -> list[SectionRecord]:
    sections: list[SectionRecord] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("JSONL input must contain one JSON object per line.")
            section = section_from_mapping(item, path.name)
            if section:
                sections.append(section)
    return sections


def sections_from_json_like(data: Any, source_file: str) -> list[SectionRecord]:
    if isinstance(data, dict):
        if isinstance(data.get("sections"), list):
            sections = [section_from_mapping(item, source_file) for item in data["sections"] if isinstance(item, dict)]
            return [section for section in sections if section is not None]

        section = section_from_mapping(data, source_file)
        return [section] if section else []

    if isinstance(data, list):
        sections = [section_from_mapping(item, source_file) for item in data if isinstance(item, dict)]
        return [section for section in sections if section is not None]

    raise ValueError("JSON input must be an object, a list of objects, or contain a 'sections' array.")


@dataclass(frozen=True)
class PdfBlock:
    text: str
    page_index: int
    min_x: float
    min_y: float
    max_y: float
    max_font: float
    font_names: tuple[str, ...]


def extract_pdf_blocks(doc: fitz.Document) -> list[PdfBlock]:
    blocks: list[PdfBlock] = []
    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue

            spans: list[dict[str, Any]] = []
            texts: list[str] = []
            for line in raw_block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        spans.append(span)
                        texts.append(span_text)

            if not texts:
                continue

            text = normalize_whitespace(" ".join(texts))
            if not text:
                continue

            blocks.append(
                PdfBlock(
                    text=text,
                    page_index=page_index,
                    min_x=float(raw_block["bbox"][0]),
                    min_y=float(raw_block["bbox"][1]),
                    max_y=float(raw_block["bbox"][3]),
                    max_font=max(float(span["size"]) for span in spans),
                    font_names=tuple(sorted({str(span["font"]) for span in spans})),
                )
            )
    return blocks


def is_pdf_noise(block: PdfBlock, page_height: float) -> bool:
    if block.min_y < 40 and re.fullmatch(r"\d{4}", block.text):
        return True
    if block.max_y > page_height - 30:
        return True
    if re.fullmatch(r"\d+", block.text):
        return True
    if "NCCN Guidelines for Patients" in block.text and block.max_font < 16:
        return True
    return False


def looks_like_breadcrumb(text: str) -> bool:
    return "»" in text or ">" in text


def split_breadcrumb(text: str) -> list[str]:
    parts = re.split(r"[»>]", text)
    return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]


def is_heading_block(block: PdfBlock, body_font: float) -> bool:
    text = block.text
    word_count = len(text.split())
    if word_count == 0:
        return False
    if looks_like_breadcrumb(text):
        return False
    if word_count > 14:
        return False
    if text.endswith((".", "!", "?")) and block.max_font <= body_font + 1.0:
        return False
    if block.max_font < max(body_font + 1.6, body_font * 1.18):
        return False
    if re.fullmatch(r"[“”\"'`-]+", text):
        return False
    return True


def build_levels_from_breadcrumb(breadcrumb: list[str], heading: str) -> tuple[str | None, str | None, str | None, str | None]:
    parts = breadcrumb[:] if breadcrumb else []
    if not parts or parts[-1] != heading:
        parts.append(heading)
    parts = parts[:4]
    padded = parts + [None] * (4 - len(parts))
    return padded[0], padded[1], padded[2], padded[3]


def load_pdf_sections(path: Path) -> list[SectionRecord]:
    doc = fitz.open(path)
    blocks = extract_pdf_blocks(doc)
    if not blocks:
        return []

    body_font_candidates = [block.max_font for block in blocks if len(block.text.split()) > 8]
    body_font = median(body_font_candidates) if body_font_candidates else 11.0

    sections: list[SectionRecord] = []
    current_heading = path.stem
    current_levels = (path.stem, None, None, None)
    current_paragraphs: list[str] = []
    current_breadcrumb: list[str] = []

    page_heights = [page.rect.height for page in doc]
    sorted_blocks = sorted(blocks, key=lambda block: (block.page_index, block.min_y, block.min_x))

    for block in sorted_blocks:
        if is_pdf_noise(block, page_heights[block.page_index]):
            continue

        if looks_like_breadcrumb(block.text) and block.min_y < 80:
            current_breadcrumb = split_breadcrumb(block.text)
            continue

        if is_heading_block(block, body_font):
            if current_paragraphs:
                section = make_section(
                    path.name,
                    None,
                    level1=current_levels[0],
                    level2=current_levels[1],
                    level3=current_levels[2],
                    level4=current_levels[3],
                    heading=current_heading,
                    paragraphs=current_paragraphs,
                )
                if section:
                    sections.append(section)
            current_heading = block.text
            current_levels = build_levels_from_breadcrumb(current_breadcrumb, current_heading)
            current_paragraphs = []
            continue

        current_paragraphs.append(block.text)

    if current_paragraphs:
        section = make_section(
            path.name,
            None,
            level1=current_levels[0],
            level2=current_levels[1],
            level3=current_levels[2],
            level4=current_levels[3],
            heading=current_heading,
            paragraphs=current_paragraphs,
        )
        if section:
            sections.append(section)

    return sections


def load_sections(path: Path, input_format: str) -> list[SectionRecord]:
    if input_format == "pdf":
        return load_pdf_sections(path)
    if input_format == "txt":
        return load_txt_sections(path)
    if input_format == "json":
        return load_json_sections(path)
    if input_format == "jsonl":
        return load_jsonl_sections(path)
    raise ValueError(f"Unsupported input format: {input_format}")


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def rolling_context_terms(sentences: list[SentenceUnit], start: int, end: int) -> set[str]:
    terms: set[str] = set()
    for index in range(max(start, end - 3), end):
        terms |= sentences[index].terms
    return terms


def should_split_before(
    sentences: list[SentenceUnit],
    chunk_start: int,
    current_end: int,
    next_index: int,
    min_words: int,
    target_words: int,
    max_words: int,
) -> bool:
    current_units = sentences[chunk_start:current_end]
    next_unit = sentences[next_index]
    current_words = sum(unit.word_count for unit in current_units)
    next_words = next_unit.word_count
    proposed_words = current_words + next_words

    if current_words >= min_words and proposed_words > max_words:
        return True

    if current_words < min_words:
        return False

    current_context = rolling_context_terms(sentences, chunk_start, current_end)
    next_terms = next_unit.terms
    similarity = jaccard_similarity(current_context, next_terms)
    paragraph_shift = next_unit.paragraph_index != sentences[current_end - 1].paragraph_index

    if current_words >= target_words and paragraph_shift:
        return True

    if current_words >= target_words and similarity < 0.08:
        return True

    if current_words >= min_words and paragraph_shift and similarity < 0.15 and proposed_words >= target_words:
        return True

    return False


def choose_overlap_start(sentences: list[SentenceUnit], chunk_start: int, chunk_end: int, overlap_sentences: int) -> int:
    if overlap_sentences <= 0:
        return chunk_end
    next_start = max(chunk_start + 1, chunk_end - overlap_sentences)
    while next_start < chunk_end - 1 and is_awkward_chunk_start(sentences[next_start]):
        next_start += 1
    return min(next_start, chunk_end)


def build_chunk_ranges(
    sentences: list[SentenceUnit],
    min_words: int,
    target_words: int,
    max_words: int,
    overlap_sentences: int,
) -> list[ChunkRange]:
    if not sentences:
        return []

    ranges: list[ChunkRange] = []
    start = 0
    total_sentences = len(sentences)

    while start < total_sentences:
        end = start + 1
        while end < total_sentences:
            if should_split_before(sentences, start, end, end, min_words, target_words, max_words):
                break
            end += 1

        while end < total_sentences:
            current_words = sum(unit.word_count for unit in sentences[start:end])
            if current_words >= min_words:
                break
            end += 1

        ranges.append(ChunkRange(start=start, end=end))
        if end >= total_sentences:
            break
        start = choose_overlap_start(sentences, start, end, overlap_sentences)

    return ranges


def trim_chunk_edges(
    sentences: list[SentenceUnit],
    chunk_range: ChunkRange,
    min_words: int,
) -> ChunkRange:
    start = chunk_range.start
    end = chunk_range.end

    while start < end - 1 and is_awkward_chunk_start(sentences[start]):
        remaining_words = sum(unit.word_count for unit in sentences[start + 1 : end])
        if remaining_words < min_words:
            break
        start += 1

    return ChunkRange(start=start, end=end)


def heading_label(section: SectionRecord) -> str:
    parts = [section.level1, section.level2, section.level3, section.level4, section.heading]
    return " > ".join(part for part in parts if part) or "(unlabeled section)"


def make_chunk_records(
    sections: list[SectionRecord],
    tokenizer: PunktSentenceTokenizer,
    min_words: int,
    target_words: int,
    max_words: int,
    overlap_sentences: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    chunk_sizes: list[int] = []
    overlap_counts: list[int] = []
    chunks_per_heading: Counter[str] = Counter()
    boundary_violations = 0

    global_chunk_id = 0
    for section in sections:
        sentences = section_to_sentences(section, tokenizer)
        if not sentences:
            continue

        base_ranges = build_chunk_ranges(sentences, min_words, target_words, max_words, overlap_sentences)
        ranges = [trim_chunk_edges(sentences, chunk_range, min_words) for chunk_range in base_ranges]

        for chunk_index, chunk_range in enumerate(ranges):
            chunk_units = sentences[chunk_range.start:chunk_range.end]
            content = render_chunk(chunk_units)
            word_count = count_words(content)
            sentence_count = len(chunk_units)
            prev_chunk_id = global_chunk_id - 1 if global_chunk_id > 0 else None

            record = {
                "chunk_id": global_chunk_id,
                "source_file": section.source_file,
                "level1": section.level1,
                "level2": section.level2,
                "level3": section.level3,
                "level4": section.level4,
                "heading": section.heading,
                "chunk_index": chunk_index,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": None,  # filled after all records are built
                "word_count": word_count,
                "sentence_count": sentence_count,
                "content": content,
            }
            records.append(record)
            chunk_sizes.append(word_count)
            chunks_per_heading[heading_label(section)] += 1
            if chunk_index > 0:
                previous_range = ranges[chunk_index - 1]
                overlap_counts.append(max(0, previous_range.end - chunk_range.start))
            global_chunk_id += 1

    for index, record in enumerate(records):
        record["prev_chunk_id"] = records[index - 1]["chunk_id"] if index > 0 else None
        record["next_chunk_id"] = records[index + 1]["chunk_id"] if index < len(records) - 1 else None

    stats = {
        "chunk_sizes": chunk_sizes,
        "chunks_per_heading": chunks_per_heading,
        "overlap_counts": overlap_counts,
        "boundary_violations": boundary_violations,
    }
    return records, stats


def validate_chunks(records: list[dict[str, Any]], max_words: int, boundary_violations: int) -> ChunkValidation:
    empty_chunks = 0
    ending_without_punctuation = 0
    starts_lowercase = 0
    starts_with_transition = 0
    oversized_chunks = 0

    for record in records:
        content = str(record.get("content", "")).strip()
        if not content:
            empty_chunks += 1
            continue

        if not content.endswith(SENTENCE_ENDINGS):
            ending_without_punctuation += 1
        if starts_lowercase_unexpectedly(content):
            starts_lowercase += 1
        if is_transition_start(content):
            starts_with_transition += 1
        if int(record.get("word_count", 0)) > int(max_words * 1.35):
            oversized_chunks += 1

    return ChunkValidation(
        empty_chunks=empty_chunks,
        ending_without_punctuation=ending_without_punctuation + boundary_violations,
        starts_lowercase_unexpectedly=starts_lowercase,
        starts_with_transition_word=starts_with_transition,
        oversized_chunks=oversized_chunks,
    )


def print_stats(
    input_path: Path,
    output_path: Path,
    input_format: str,
    section_count: int,
    chunk_records: list[dict[str, Any]],
    stats: dict[str, Any],
    validation: ChunkValidation,
) -> None:
    chunk_sizes: list[int] = stats["chunk_sizes"]
    chunks_per_heading: Counter[str] = stats["chunks_per_heading"]
    overlap_counts: list[int] = stats["overlap_counts"]

    avg_words = mean(chunk_sizes) if chunk_sizes else 0.0
    min_words = min(chunk_sizes) if chunk_sizes else 0
    max_words = max(chunk_sizes) if chunk_sizes else 0
    avg_overlap = mean(overlap_counts) if overlap_counts else 0.0

    print(f"Input file: {input_path}")
    print(f"Input format: {input_format}")
    print(f"Output file: {output_path}")
    print(f"Sections processed: {section_count}")
    print(f"Total chunks: {len(chunk_records)}")
    print(f"Average chunk size: {avg_words:.1f}")
    print(f"Min chunk size: {min_words}")
    print(f"Max chunk size: {max_words}")
    print(f"Average overlap sentences: {avg_overlap:.2f}")
    print(f"Chunks with overlap: {len(overlap_counts)}")
    print("Validation:")
    print(f"  Empty chunks: {validation.empty_chunks}")
    print(f"  Chunks ending without punctuation: {validation.ending_without_punctuation}")
    print(f"  Chunks starting lowercase unexpectedly: {validation.starts_lowercase_unexpectedly}")
    print(f"  Chunks starting with transition words: {validation.starts_with_transition_word}")
    print(f"  Oversized chunks: {validation.oversized_chunks}")
    print("Chunks per heading:")
    for label, count in sorted(chunks_per_heading.items()):
        print(f"  {count:>3}  {label}")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, str, int, int, int, int]:
    if args.min_words <= 0:
        raise ValueError("--min-words must be greater than 0")
    if args.target_words <= 0:
        raise ValueError("--target-words must be greater than 0")
    if args.max_words <= 0:
        raise ValueError("--max-words must be greater than 0")
    if not (args.min_words <= args.target_words <= args.max_words):
        raise ValueError("Expected --min-words <= --target-words <= --max-words")
    if args.overlap_sentences < 0:
        raise ValueError("--overlap-sentences cannot be negative")

    input_path = resolve_input_path(args.input)
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output
    input_format = detect_input_format(input_path, args.input_format)
    return (
        input_path,
        output_path,
        input_format,
        args.min_words,
        args.target_words,
        args.max_words,
        args.overlap_sentences,
    )


def main() -> None:
    args = parse_args()
    (
        input_path,
        output_path,
        input_format,
        min_words,
        target_words,
        max_words,
        overlap_sentences,
    ) = validate_args(args)

    tokenizer = build_sentence_tokenizer()
    sections = load_sections(input_path, input_format)
    records, stats = make_chunk_records(
        sections=sections,
        tokenizer=tokenizer,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
        overlap_sentences=overlap_sentences,
    )
    validation = validate_chunks(records, max_words, stats["boundary_violations"])
    write_jsonl(records, output_path)
    print_stats(
        input_path=input_path,
        output_path=output_path,
        input_format=input_format,
        section_count=len(sections),
        chunk_records=records,
        stats=stats,
        validation=validation,
    )


if __name__ == "__main__":
    main()
