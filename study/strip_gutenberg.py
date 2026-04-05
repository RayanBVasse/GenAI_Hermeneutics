"""
Strip Project Gutenberg headers and footers from downloaded text files.
Keeps only the content between *** START ... *** and *** END ... *** markers.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def strip_gutenberg(file_path: Path) -> None:
    """Strip Gutenberg boilerplate in-place."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if "*** START OF" in line.upper():
            start_idx = i + 1
        elif "*** END OF" in line.upper():
            end_idx = i
            break

    content = "\n".join(lines[start_idx:end_idx]).strip()
    file_path.write_text(content, encoding="utf-8")
    print(f"Stripped {file_path.name}: {len(lines)} -> {end_idx - start_idx} lines")


def main():
    books_dir = PROJECT_ROOT / "data" / "books"
    for txt_file in sorted(books_dir.glob("*.txt")):
        # Only strip Gutenberg files (check for marker)
        text = txt_file.read_text(encoding="utf-8")
        if "PROJECT GUTENBERG" in text.upper():
            strip_gutenberg(txt_file)
        else:
            print(f"Skipping {txt_file.name} (not a Gutenberg file)")


if __name__ == "__main__":
    main()
