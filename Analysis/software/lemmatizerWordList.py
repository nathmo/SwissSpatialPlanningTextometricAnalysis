import os
import treetaggerwrapper

# --- Configuration ---
TAGGER_DIR = '/home/nemo/.TXM-0.8.4/plugins/org.txm.treetagger.core.linux_1.0.0.202505130851/linux'
TAGGER_PAR_FR = '/home/nemo/.TXM-0.8.4/plugins/org.txm.treetagger.core.models_1.0.0.202505130851/models/fr.par'
TAGGER_PAR_DE = '/home/nemo/.TXM-0.8.4/plugins/org.txm.treetagger.core.models_1.0.0.202505130851/models/german.par'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "lemmatized_word_lists"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Cached taggers to avoid reloading models repeatedly ---
taggers = {}

def get_tagger(lang: str):
    """Return a cached TreeTagger instance for the given language."""
    if lang not in taggers:
        if lang == 'DE':
            taggers['DE'] = treetaggerwrapper.TreeTagger(
                TAGLANG='de',
                TAGDIR=TAGGER_DIR,
                TAGPARFILE=TAGGER_PAR_DE
            )
        elif lang == 'FR':
            taggers['FR'] = treetaggerwrapper.TreeTagger(
                TAGLANG='fr',
                TAGDIR=TAGGER_DIR,
                TAGPARFILE=TAGGER_PAR_FR
            )
        else:
            raise ValueError(f"Unsupported language: {lang}")
    return taggers[lang]


def lemmatize_line(line: str, tagger) -> str:
    """Lemmatize a single line of text and print word→lemma replacements."""
    line = line.strip()
    if not line:
        return ""
    tags = tagger.tag_text(line)
    lemmas = []
    for t in tags:
        parts = t.split('\t')
        if len(parts) == 3:
            word, _, lemma = parts
            lemma = lemma.lower()
            if lemma == '<unknown>':
                lemma = word.lower()
            elif lemma != word:
                print(f"{word} -> {lemma}")
            lemmas.append(lemma.lower())
        else:
            # Fallback if tagger output doesn't have 3 fields
            lemmas.append(parts[0].lower())
    return " ".join(lemmas)



def lemmatize_file(input_path: str, output_path: str, lang: str):
    """Lemmatize a text file line by line using the correct tagger."""
    tagger = get_tagger(lang)
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(lemmatize_line(line, tagger) + "\n")


def process_folder(folder_path: str):
    """Process all .txt files in a folder, selecting language by filename."""
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".txt"):
            continue

        # Determine language from filename
        if filename.endswith("DE.txt"):
            lang = 'DE'
        elif filename.endswith("FR.txt"):
            lang = 'FR'
        else:
            print(f"Language not supported, skipping {filename}")
            break
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        print("-------------------------------------------------------------------")
        print(f"Lemmatizing ({lang}): {filename} → {os.path.basename(output_path)}")
        lemmatize_file(input_path, output_path, lang)


# --- Run ---
print("=== Processing corpus ===")
process_folder(DATA_FOLDER)
print("✅ Lemmatization complete.")
