import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "FR"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "pos_lists"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Reference word lists
WORDLIST_FILES = [
    os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesMotiliteDE.txt")),
    os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsDE.txt"))
]

def sort_and_deduplicate_file(input_path, output_path):
    """Read lines, strip whitespace, remove duplicates, sort alphabetically, and save."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Remove duplicates (case-insensitive)
    unique_lines = list({line.casefold(): line for line in lines}.values())

    # Sort alphabetically (case-insensitive)
    sorted_lines = sorted(unique_lines, key=str.casefold)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_lines) + "\n")

    print(f"✅ Sorted and deduplicated: {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def main():
    for file_path in WORDLIST_FILES:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            sort_and_deduplicate_file(file_path, output_path)
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    main()
