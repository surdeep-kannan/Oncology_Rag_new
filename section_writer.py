from pathlib import Path


OUTPUT_FILE = Path("aya_sections.txt")


def main() -> None:
    print("\n=== NCCN Section Writer ===")

    while True:
        print("\nChoose heading type:")
        print("1 = Level 1")
        print("2 = Level 2")
        print("3 = Level 3")
        print("q = Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "q":
            break

        if choice not in {"1", "2", "3"}:
            print("Invalid choice")
            continue

        heading = input("Heading: ").strip()

        print("\nPaste content below.")
        print("Type END on a new line when finished.\n")

        content_lines: list[str] = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            content_lines.append(line)

        content = "\n".join(content_lines).strip()

        with OUTPUT_FILE.open("a", encoding="utf-8") as file:
            if choice == "1":
                file.write(f"\n=== LEVEL1: {heading} ===\n\n")
            elif choice == "2":
                file.write(f"\n=== LEVEL2: {heading} ===\n\n")
            else:
                file.write(f"\n=== LEVEL3: {heading} ===\n\n")

            file.write(content + "\n")

        print(f"\nSaved section: {heading}")

    print(f"\nAll sections saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
