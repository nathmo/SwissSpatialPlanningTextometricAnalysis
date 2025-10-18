import os
import treetaggerwrapper

tagger = treetaggerwrapper.TreeTagger(
    TAGLANG='fr',
    TAGDIR='/home/nemo/.TXM-0.8.4/plugins/org.txm.treetagger.core.linux_1.0.0.202505130851/linux',
    TAGPARFILE='/home/nemo/.TXM-0.8.4/plugins/org.txm.treetagger.core.models_1.0.0.202505130851/models/fr.par'
)

# --- Path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "pos_lists"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Initialize TreeTagger (ensure TreeTagger installed + french parameter file available) ---

def lemmatize_line(line: str) -> str:
    """Lemmatize a single line of text (space-separated words)."""
    if not line.strip():
        return ""
    tags = tagger.tag_text(line.strip())
    lemmas = []
    for t in tags:
        # Each tag is like "mot\tPOS\tlemme"
        parts = t.split('\t')
        if len(parts) == 3:
            _, _, lemma = parts
            # TreeTagger sometimes returns '<unknown>' if lemma unknown
            lemmas.append(lemma if lemma != '<unknown>' else parts[0])
        else:
            lemmas.append(parts[0])
    return " ".join(lemmas)

# --- Process all .txt files in dataset ---
for filename in os.listdir(DATA_FOLDER):
    if filename.lower().endswith(".txt"):
        input_path = os.path.join(DATA_FOLDER, filename)
        output_path = os.path.join(
            OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_lemmatized.txt"
        )

        print(f"Lemmatizing: {filename} → {os.path.basename(output_path)}")

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                lemmatized = lemmatize_line(line)
                fout.write(lemmatized + "\n")

print("✅ Lemmatization complete.")
