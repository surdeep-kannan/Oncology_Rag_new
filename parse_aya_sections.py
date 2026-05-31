import argparse
import json
import re
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
LEVEL_RE = re.compile(r"^===\s*LEVEL([123]):\s*(.*?)\s*===\s*$")


def clean_content(text: str) -> str:
    text = text.replace("\uFFFE", "").replace("\uFFFD", "")
    text = text.replace("￾", "")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n(?=[a-z])", " ", text)
    text = re.sub(r"\n(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_heading_and_content(
    level1: Optional[str],
    level2: Optional[str],
    level3: Optional[str],
    content: str,
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    active_heading = level3 or level2 or level1
    if not active_heading:
        return level1, level2, level3, content

    heading_match = re.match(r"^(.*\b(before|after|during|with|without|for|of))\s*$", active_heading, re.IGNORECASE)
    content_match = re.match(r"^([a-z][a-z/-]*)\s+([A-Z].*)$", content)
    if heading_match and content_match:
        extra_word = content_match.group(1)
        repaired_content = content_match.group(2).strip()
        repaired_heading = f"{active_heading} {extra_word}".strip()
        if level3:
            level3 = repaired_heading
        elif level2:
            level2 = repaired_heading
        else:
            level1 = repaired_heading
        return level1, level2, level3, repaired_content

    return level1, level2, level3, content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(BASE_DIR / "aya_sections.txt"))
    parser.add_argument("--output", default=str(BASE_DIR / "aya_sections_chunks.jsonl"))
    return parser.parse_args()


def build_record(
    level1: Optional[str],
    level2: Optional[str],
    level3: Optional[str],
    content_lines: list[str],
    output: list[dict],
) -> None:
    content = "\n".join(line.rstrip() for line in content_lines).strip()
    content = clean_content(content)
    if not content:
        return

    level1, level2, level3, content = repair_heading_and_content(level1, level2, level3, content)

    heading = level3 or level2 or level1
    output.append(
        {
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "level4": None,
            "heading": heading,
            "content": content,
        }
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    level1: Optional[str] = None
    level2: Optional[str] = None
    level3: Optional[str] = None
    current_content: list[str] = []
    records: list[dict] = []

    for line in lines:
        match = LEVEL_RE.match(line.strip())
        if match:
            build_record(level1, level2, level3, current_content, records)
            current_content = []

            level_num = match.group(1)
            heading = match.group(2).strip()

            if level_num == "1":
                level1 = heading
                level2 = None
                level3 = None
            elif level_num == "2":
                level2 = heading
                level3 = None
            else:
                level3 = heading

            continue

        current_content.append(line)

    build_record(level1, level2, level3, current_content, records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Parsed {len(records)} sections")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
