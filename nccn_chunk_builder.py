import json
from pathlib import Path


OUTPUT_FILE = Path("aya_nccn_chunks.jsonl")


def main() -> None:
    print("\n=== NCCN Chunk Builder ===\n")

    level1 = input("Level 1: ").strip()
    level2 = input("Level 2: ").strip()

    while True:
        print("\n--- New Chunk ---")

        level3 = input("Level 3 Heading (or type 'exit'): ").strip()

        if level3.lower() == "exit":
            break

        print("\nPaste content below.")
        print("When finished, type: END\n")

        lines: list[str] = []

        while True:
            line = input()

            if line.strip() == "END":
                break

            lines.append(line)

        content = "\n".join(lines).strip()

        chunk = {
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "content": content,
        }

        with OUTPUT_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"\nSaved chunk: {level3}")

    print(f"\nAll chunks saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
