import os
from lxml import etree
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
WORDLIST_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsMotilité.txt"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "pos_lists"))

WINDOW_MULTI = 3  # window for multi-word expressions

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_lemmas_and_pos_from_file(file_path):
    """Extract (lemma, POS) pairs from a TEI XML file."""
    lemmas = []
    pos_list = []
    try:
        tree = etree.parse(file_path)
        ns = {
            "tei": "http://www.tei-c.org/ns/1.0",
            "txm": "http://textometrie.org/1.0"
        }
        w_elements = tree.xpath("//tei:w", namespaces=ns)
        for w in w_elements:
            lemma_elem = w.xpath("txm:ana[@type='#frlemma']", namespaces=ns)
            pos_elem = w.xpath("txm:ana[@type='#frpos']", namespaces=ns)

            if lemma_elem and lemma_elem[0].text:
                lemma = lemma_elem[0].text.strip()
                pos = None
                if pos_elem and pos_elem[0].text:
                    pos = pos_elem[0].text.split(":")[0]  # keep main POS tag
                lemmas.append(lemma)
                pos_list.append(pos if pos else "UNK")

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return lemmas, pos_list


# --- 1. EXTRACT ALL LEMMAS + POS ---
all_lemmas = []
all_pos = []

xml_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".xml")]
for filename in tqdm(xml_files, desc="Processing XML files"):
    path = os.path.join(DATA_FOLDER, filename)
    file_lemmas, file_pos = extract_lemmas_and_pos_from_file(path)
    all_lemmas.extend(file_lemmas)
    all_pos.extend(file_pos)

print(f"\nTotal tokens extracted: {len(all_lemmas)}")

# --- SUMMARY STATISTICS ---
pos_counts = Counter(all_pos)
print("\n=== POS CATEGORY SUMMARY ===")
for pos, count in pos_counts.most_common():
    print(f"{pos:>6} : {count}")
print("============================\n")

# ---  DETAILED POS BREAKDOWN ---
# Map POS → list of lemmas
pos_to_lemmas = defaultdict(list)
for lemma, pos in zip(all_lemmas, all_pos):
    pos_to_lemmas[pos].append(lemma)

print("=== POS CATEGORY BREAKDOWN ===")
summary_rows = []
for pos, lemmas in pos_to_lemmas.items():
    lemma_freq = Counter(lemmas)
    total_tokens = sum(lemma_freq.values())
    unique_lemmas = len(lemma_freq)

    # Sort lemmas by frequency
    most_common = [f"{lemma}({freq})" for lemma, freq in lemma_freq.most_common(5)]
    least_common = [f"{lemma}({freq})" for lemma, freq in lemma_freq.most_common()[-5:]]

    print(f"\nPOS: {pos}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Unique lemmas: {unique_lemmas}")
    print(f"  Top 5 lemmas: {', '.join(most_common)}")
    print(f"  Last 5 lemmas: {', '.join(least_common)}")

    summary_rows.append({
        "POS": pos,
        "total_tokens": total_tokens,
        "unique_lemmas": unique_lemmas,
        "top_5": ", ".join(most_common),
        "last_5": ", ".join(least_common)
    })

# --- 2. ORGANIZE LEMMAS BY POS ---
pos_to_lemmas = defaultdict(list)
for lemma, pos in zip(all_lemmas, all_pos):
    pos_to_lemmas[pos].append(lemma)

# --- 3. SAVE EACH POS AS A .TXT FILE ---
for pos, lemmas in pos_to_lemmas.items():
    unique_sorted = sorted(set(lemmas))
    output_path = os.path.join(OUTPUT_FOLDER, f"{pos}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_sorted))
    print(f"Saved {len(unique_sorted)} lemmas to {output_path}")

# --- 4. LOAD REFERENCE WORD LIST ---
if os.path.exists(WORDLIST_FILE):
    with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
        motilite_words = [line.strip() for line in f if line.strip()]
else:
    print(f"⚠️ File not found: {WORDLIST_FILE}")
    motilite_words = []

# Separate single words from multi-word expressions
single_words = [w for w in motilite_words if " " not in w]
multi_expressions = [w for w in motilite_words if " " in w]

# --- 5A. COUNT OCCURRENCES FOR SINGLE WORDS ---
lemma_counter = Counter(all_lemmas)
results = {}

for word in single_words:
    results[word] = lemma_counter[word]

# --- 5B. COUNT OCCURRENCES FOR MULTI-WORD EXPRESSIONS ---
# Each multi-word expression is matched if all its words appear within ±3 tokens.
for expr in tqdm(multi_expressions, desc="Searching multi-word expressions"):
    tokens = expr.split()
    count = 0

    for i in range(len(all_lemmas)):
        # build a window around position i
        start = max(0, i - WINDOW_MULTI)
        end = min(len(all_lemmas), i + WINDOW_MULTI + 1)
        window = all_lemmas[start:end]

        # check if all tokens in the expression are in this window
        if all(tok in window for tok in tokens):
            count += 1

    results[expr] = count

# --- 6. DISPLAY RESULTS ---
print("\n=== Word Occurrence Counts (single + multi-word) ===")
for word, count in results.items():
    print(f"{word:40} : {count}")
print("====================================================\n")
