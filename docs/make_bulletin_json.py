import json
import re
from pathlib import Path

# Set the file path to the markdown file
input_md_file = Path(__file__).resolve().parent / "bulletin.md"

# Regex for date
date_pattern = r'^_(\d{2}/\d{2}/\d{4})_$'

def generate_json_file():
    with open(input_md_file, "r", encoding="utf-8") as md_file:
        markdown_content = md_file.read()
    bulletin_json = []
    content = ""

    # Initialize title and date with default values
    title = "DEFAULT_TITLE"
    date = "DEFAULT_DATE"

    for line in markdown_content.split("\n"):
        # Skip if the line begins with #
        if line.startswith("# "):
            continue
        elif line.startswith("---"):
            bulletin_json.append({"title": title, "date": date, "content": content})
            content = ""
            # Reset title and date to their default values after each section
            title = "DEFAULT_TITLE"
            date = "DEFAULT_DATE"
        elif line.startswith("## "):
            title = line[3:].strip()
        elif re.match(date_pattern, line):
            date = line[1 : len(line) - 1].strip()
        else:
            content += line + "\n"
    # Append last section
    bulletin_json.append({"title": title, "date": date, "content": content})

    with open("_static/bulletin.json", "w") as json_file:
        json.dump(bulletin_json, json_file, indent=4)


if __name__ == "__main__":
    generate_json_file()
