import os
from lxml import etree
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "FR"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "pos_lists"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Reference word lists
WORDLIST_FILES = [
    os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesMotilite.txt")),
    os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTemps.txt"))
]

WINDOW_MULTI = 3   # For multi-word expressions
KWIC_WINDOW = 20   # For KWIC context

# --- FUNCTIONS ---
def extract_lemmas_and_pos_from_file(file_path):
    """Extract (lemma, POS) pairs from a TEI XML file, plus original token."""
    lemmas = []
    pos_list = []
    tokens = []
    try:
        tree = etree.parse(file_path)
        ns = {"tei": "http://www.tei-c.org/ns/1.0", "txm": "http://textometrie.org/1.0"}
        w_elements = tree.xpath("//tei:w", namespaces=ns)
        for w in w_elements:
            # Extract raw token text
            token_elem = w.xpath("txm:form", namespaces=ns)
            token = token_elem[0].text.strip() if token_elem and token_elem[0].text else ""
            tokens.append(token)

            # Extract lemma
            lemma_elem = w.xpath("txm:ana[@type='#frlemma']", namespaces=ns)
            lemma = lemma_elem[0].text.strip() if lemma_elem and lemma_elem[0].text else token

            # Extract POS
            pos_elem = w.xpath("txm:ana[@type='#frpos']", namespaces=ns)
            pos = pos_elem[0].text.split(":")[0] if pos_elem and pos_elem[0].text else "UNK"

            lemmas.append(lemma)
            pos_list.append(pos)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return lemmas, pos_list, tokens


# --- 1. EXTRACT DATA FROM XML ---
all_lemmas = []
all_pos = []
all_tokens = []

xml_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".xml")]
for filename in tqdm(xml_files, desc="Processing XML files"):
    path = os.path.join(DATA_FOLDER, filename)
    file_lemmas, file_pos, file_tokens = extract_lemmas_and_pos_from_file(path)
    all_lemmas.extend(file_lemmas)
    all_pos.extend(file_pos)
    all_tokens.extend(file_tokens)

print(f"\nTotal raw tokens present: {len(all_tokens)}")

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

meaningful_pos = {"NOM", "VER", "ADJ", "ADV", "INT"}
filtered_lemmas_with_pos = [ lemma for lemma, pos in zip(all_lemmas, all_pos) if pos in meaningful_pos ]

print(f"\nTotal useful lemmas extracted: {len(filtered_lemmas_with_pos)}")

# --- 2. LOAD REFERENCE WORD LISTS ---
ref_words = {}
for filepath in WORDLIST_FILES:
    key = os.path.splitext(os.path.basename(filepath))[0]
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        ref_words[key] = words
    else:
        print(f"⚠️ File not found: {filepath}")
        ref_words[key] = []

# --- 3. COUNT OCCURRENCES & BUILD KWIC ---
results_counts = []
results_kwic = []

for list_name, words_list in ref_words.items():
    single_words = [w for w in words_list if " " not in w]
    multi_words = [w for w in words_list if " " in w]

    lemma_counter = Counter(all_lemmas)

    # Single word counts + KWIC
    for word in single_words:
        count = lemma_counter[word]
        results_counts.append({"list": list_name, "lemma": word, "count": count})

        # KWIC
        for i, lemma in enumerate(all_lemmas):
            if lemma == word:
                before = " ".join(all_tokens[max(0, i-KWIC_WINDOW):i])
                hit = all_tokens[i]
                after = " ".join(all_tokens[i+1:i+1+KWIC_WINDOW])
                results_kwic.append({
                    "lemma": word,
                    "before": before,
                    "hit": hit,
                    "after": after,
                    "list": list_name
                })
    print("Done searching for single word expression")
    # Multi-word counts + KWIC
    for expr in tqdm(multi_words, desc=f"Searching multi-word expressions ({list_name})"):
        tokens_expr = expr.split()
        count = 0
        for i in range(len(all_lemmas)):
            start = max(0, i - WINDOW_MULTI)
            end = min(len(all_lemmas), i + WINDOW_MULTI + 1)
            window = all_lemmas[start:end]

            if all(tok in window for tok in tokens_expr):
                count += 1

                # KWIC for multi-word expression
                before = " ".join(all_tokens[max(0, i-KWIC_WINDOW):i])
                hit = " ".join(all_tokens[i:i+len(tokens_expr)])
                after = " ".join(all_tokens[i+len(tokens_expr):i+len(tokens_expr)+KWIC_WINDOW])
                results_kwic.append({
                    "lemma": expr,
                    "before": before,
                    "hit": hit,
                    "after": after,
                    "list": list_name
                })
        results_counts.append({"list": list_name, "lemma": expr, "count": count})

# --- 4. EXPORT RESULTS ---
counts_df = pd.DataFrame(results_counts)
kwic_df = pd.DataFrame(results_kwic)

counts_csv = os.path.join(OUTPUT_FOLDER, "word_occurrences_counts.csv")
kwic_csv = os.path.join(OUTPUT_FOLDER, "word_occurrences_kwic.csv")

counts_df.to_csv(counts_csv, index=False, encoding="utf-8")
kwic_df.to_csv(kwic_csv, index=False, encoding="utf-8")

print(f"✅ Occurrence counts saved to {counts_csv}")
print(f"✅ KWIC saved to {kwic_csv}")
