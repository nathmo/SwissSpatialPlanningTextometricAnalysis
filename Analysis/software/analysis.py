import os
from lxml import etree
from collections import Counter, defaultdict
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
from adjustText import adjust_text
# ----------------------
# CONFIGURATION
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDERS = {
    "FR": os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "FR")),
    "DE": os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "DE"))
}
OUTPUT_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "result"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WORDLIST_FILES = {
    "FR": [
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesAccessibilitéFR.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesMotiliteFR.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsChronoFR.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsVecueFR.txt"))
    ],
    "DE": [
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesAccessibilitéDE.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesMotiliteDE.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsChronoDE.txt")),
        os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "lemmesTempsVecueDE.txt"))
    ]
}

KWIC_WINDOW = 50
WINDOW_MULTI = 2  # max words allowed between multi-word reference words


# ----------------------
# PART 1: LOAD XML DATA
# ----------------------
def parse_xml_file(file_path, lang):
    """Parse a single XML file and return list of word dictionaries."""
    ns = {"tei": "http://www.tei-c.org/ns/1.0", "txm": "http://textometrie.org/1.0"}
    words = []
    if lang == "DE":
        codeXMLlanguage = "german"
    else:
        codeXMLlanguage = "fr"
    try:
        tree = etree.parse(file_path)
        w_elements = tree.xpath("//tei:w", namespaces=ns)
        for w in w_elements:
            w_id = w.get("id")  # e.g., w_FR_BE_2002
            n = int(w.get("n"))
            document_name = "_".join(w_id.split("_")[1:3])
            language = w_id.split("_")[1]
            canton = w_id.split("_")[2]
            year = int(w_id.split("_")[3])

            form_elem = w.xpath("txm:form", namespaces=ns)
            form = form_elem[0].text.strip() if form_elem and form_elem[0].text else ""

            pos_elem = w.xpath(f"txm:ana[@type='#{codeXMLlanguage.lower()}pos']", namespaces=ns)
            lemma_elem = w.xpath(f"txm:ana[@type='#{codeXMLlanguage.lower()}lemma']", namespaces=ns)

            pos = pos_elem[0].text.strip() if pos_elem and pos_elem[0].text else "UNK"
            lemma = lemma_elem[0].text.strip() if lemma_elem and lemma_elem[0].text else form

            words.append({
                "id": w_id,
                "document": document_name,
                "canton": canton,
                "year": year,
                "n": n,
                "form": form,
                f"{lang.lower()}pos": pos,
                f"{lang.lower()}lemma": lemma,
                "lang": lang
            })
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return words


def load_dataset(lang):
    folder = DATA_FOLDERS[lang]
    xml_files = [f for f in os.listdir(folder) if f.endswith(".xml")]
    dataset = []
    for filename in tqdm(xml_files, desc=f"Loading {lang} files"):
        file_path = os.path.join(folder, filename)
        dataset.extend(parse_xml_file(file_path, lang))
    # count UNK
    unk_count = sum(1 for w in dataset if w[f"{lang.lower()}pos"] == "UNK")
    if unk_count > 0:
        print(f"⚠️ {unk_count} words with UNK {lang} POS")
    return dataset


# ----------------------
# PART 2: POS STATISTICS
# ----------------------
def pos_statistics(dataset, lang):
    """
    Compute POS statistics, save per-POS CSVs, and summary CSV.
    """
    pos_key = f"{lang.lower()}pos"
    lemma_key = f"{lang.lower()}lemma"

    pos_to_lemmas = defaultdict(list)
    for w in dataset:
        pos_to_lemmas[w[pos_key]].append(w[lemma_key])

    summary_stats = []
    total_lemma_count = 0
    total_unique_lemmas_set = set()

    for pos, lemmas in pos_to_lemmas.items():
        lemma_counter = Counter(lemmas)
        total_tokens = sum(lemma_counter.values())
        unique_lemmas = len(lemma_counter)

        total_lemma_count += total_tokens
        total_unique_lemmas_set.update(lemma_counter.keys())

        # Save CSV per POS
        csv_path = os.path.join(OUTPUT_FOLDER, f"{lang}_{pos}_lemmas.csv")
        pd.DataFrame(lemma_counter.items(), columns=["lemma", "count"])\
          .sort_values("count", ascending=False)\
          .to_csv(csv_path, index=False, encoding="utf-8")

        summary_stats.append({
            "POS": pos,
            "total_tokens": total_tokens,
            "unique_lemmas": unique_lemmas
        })

    # Save summary CSV
    summary_csv_path = os.path.join(OUTPUT_FOLDER, f"{lang}_POS_summary.csv")
    pd.DataFrame(summary_stats)\
      .sort_values("total_tokens", ascending=False)\
      .to_csv(summary_csv_path, index=False, encoding="utf-8")

    print(f"POS statistics for {lang}:")
    for stat in summary_stats:
        print(stat)

    print(f"\nTotal lemma occurrences in dataset: {total_lemma_count}")
    print(f"Total unique lemmas in dataset: {len(total_unique_lemmas_set)}")

    return pos_to_lemmas


# ----------------------
# PART 3: FILTER DATASET BY WORDLIST
# ----------------------
def load_wordlist(lang):
    words = {}
    for path in WORDLIST_FILES[lang]:
        key = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                words[key] = [line.strip() for line in f if line.strip()]
        else:
            words[key] = []
            print(f"⚠️ File not found: {path}")
    return words


def match_multiword(lemmas, ref_expr):
    """Return index positions where multiword expression matches with up to WINDOW_MULTI words in between."""
    ref_tokens = ref_expr.split()
    matches = []
    i = 0
    while i <= len(lemmas) - len(ref_tokens):
        idx_list = [i]
        j = 0
        k = i
        success = True
        for token in ref_tokens:
            found = False
            for offset in range(WINDOW_MULTI + 1):
                if k + offset < len(lemmas) and lemmas[k + offset] == token:
                    k = k + offset + 1
                    idx_list.append(k - 1)
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            matches.append(idx_list[:-1])
            i = idx_list[-1] + 1
        else:
            i += 1
    return matches





def filter_dataset(dataset, lang, wordlists):
    """
    Filter dataset by wordlists (single and multiword) and also compute lemma occurrence counts.

    Returns:
        filtered: list of dicts for matched tokens (like before)
        lemma_counts: list of dicts {"lemma": ..., "list": ..., "count": ...} for CSV export
    """
    lemma_key = f"{lang.lower()}lemma"
    filtered = []
    lemmas = [w[lemma_key].lower() for w in dataset]

    # Counter for lemma occurrences
    lemma_counter = Counter()

    # Outer tqdm for lists
    for list_name, expressions in wordlists.items():
        # Inner tqdm for expressions
        for expr in tqdm(expressions, desc=f"Filtering {lang} wordlists, {list_name} "):
            if " " not in expr:  # single word
                for w in dataset:
                    if w[lemma_key] == expr:
                        w_copy = w.copy()
                        w_copy["list"] = list_name
                        filtered.append(w_copy)
                        lemma_counter[(expr, list_name)] += 1
            else:  # multiword
                matches = match_multiword(lemmas, expr)
                for idx_list in matches:
                    hit_words = [dataset[i].copy() for i in idx_list]
                    for hw in hit_words:
                        hw["list"] = list_name
                        filtered.append(hw)
                    # Count the multiword occurrence only once
                    lemma_counter[(expr, list_name)] += 1

    # Convert counter to a list of dicts for CSV export
    lemma_counts = [{"lemma": lemma, "list": list_name, "count": count}
                    for (lemma, list_name), count in lemma_counter.items()]

    return filtered, lemma_counts

# ----------------------
# PART 4: KWIC GENERATION
# ----------------------
def generate_kwic(dataset_filtered, dataset, lang):
    """Generate KWIC (keyword-in-context) rows from dataset_filtered."""

    kwic_rows = []
    for w in dataset_filtered:
        n = w["n"]
        # Safely get preceding and following context
        before_context = dataset[max(0, n - KWIC_WINDOW-1):n]
        after_context = dataset[n:n  + KWIC_WINDOW]

        before = " ".join(x["form"] for x in before_context)
        hit = w["form"]
        after = " ".join(x["form"] for x in after_context)

        kwic_rows.append({
            "before": before,
            "hit": hit,
            "after": after,
            "lemma": w.get(f"{lang.lower()}lemma", w.get("lemma", "")),
            "list": w["list"],
            "n": n,
            "lang": w["lang"],
            "document": w["document"],
            "year": w["year"],
            "canton": w["canton"]
        })

    return kwic_rows



# ----------------------
# PART 5: PCA / AFC
# ----------------------

def build_matrix(kwic_rows, lang, by="document"):
    """Create lemma vs document/year/canton matrix, filtered by language."""
    rows = defaultdict(lambda: defaultdict(int))

    for w in kwic_rows:
        # Filter to only include rows matching the requested language
        #print(w.get("lang", "").lower())
        if w.get("lang", "").lower() == lang.lower():
            key = w[by]
            rows[key][w["lemma"]] += 1

    if not rows:
        print(f"⚠️ No data found for language '{lang}' in kwic_rows.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).fillna(0)
    # Normalize per column
    df = df.div(df.sum(axis=0), axis=1)
    return df



def run_pca(kwic_rows, lang, top_words=20, cutoff_radius=0.075, highlight_words=None):
    """
    Run PCA on lemma frequency matrices with options for:
      - cutoff for words near the center (radius around 0)
      - highlight specific words in green (hypothetical single-word documents)
    """

    if highlight_words is None:
        highlight_words = [
            ""#"TP", "TIM", "IV"
        ]
    highlight_words = [w for w in highlight_words]

    def _pca_and_plot(matrix_df, label_suffix, by):
        n_components = min(10, matrix_df.shape[0], matrix_df.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(matrix_df.T)

        total_var = pca.explained_variance_ratio_.sum()
        var_pc1 = pca.explained_variance_ratio_[0]
        var_pc2 = pca.explained_variance_ratio_[1]
        print(f"\n--- PCA ({lang}) {label_suffix} ---")
        print(f"Total explained variance (first {n_components} PCs): {total_var:.4f}")
        for i, v in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {v*100:.2f}%")

        doc_coords = pd.DataFrame(coords, index=matrix_df.columns,
                                  columns=[f"PC{i+1}" for i in range(n_components)])
        loadings = pd.DataFrame(pca.components_.T, index=matrix_df.index,
                                columns=[f"PC{i+1}" for i in range(n_components)])

        # --- Top words ---
        word_counts = matrix_df.sum(axis=1)
        top_words_idx = word_counts.sort_values(ascending=False).head(top_words).index
        freq_scaled = np.log1p(word_counts.loc[top_words_idx])
        freq_scaled = 5 + 10 * (freq_scaled - freq_scaled.min()) / (freq_scaled.max() - freq_scaled.min())

        # --- Plot ---
        plt.figure(figsize=(9, 6))
        plt.scatter(doc_coords["PC1"], doc_coords["PC2"], alpha=0.6,
                    label=f"{by.capitalize()}s", color="steelblue")

        texts = []  # collect all text objects

        # Documents
        for label in doc_coords.index:
            x, y = doc_coords.loc[label, "PC1"], doc_coords.loc[label, "PC2"]
            texts.append(plt.text(x, y, label, fontsize=8, alpha=0.7, color="blue"))

        # Top words (red), apply cutoff radius
        for word, size in zip(top_words_idx, freq_scaled):
            x, y = loadings.loc[word, "PC1"], loadings.loc[word, "PC2"]
            if np.hypot(x, y) >= cutoff_radius:
                plt.scatter(x, y, color="red", s=20)
                texts.append(plt.text(x + 0.005, y + 0.005, word, fontsize=size, color="red", alpha=0.8))

        # Hypothetical single-word documents (green)
        for word in highlight_words:
            #print("------------------------------")
            #print(matrix_df.index)
            #print("------------------------------")
            if word in matrix_df.index:
                v = np.zeros(matrix_df.shape[0])
                idx = matrix_df.index.get_loc(word)
                v[idx] = 1
                coord = pca.transform(v.reshape(1, -1))
                x, y = coord[0, 0], coord[0, 1]
                plt.scatter(x, y, color="green", s=40, alpha=0.8)
                texts.append(plt.text(x + 0.005, y + 0.005, word, fontsize=8, color="green", alpha=0.9))

        # --- Adjust text to prevent overlap ---
        adjust_text(texts, only_move={'points': 'y', 'texts': 'xy'},
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))

        # --- Axes, grid, and 0 lines ---
        plt.title(f"PCA ({lang}) {label_suffix}")
        plt.xlabel(f"PC1 ({var_pc1*100:.1f}% var)")
        plt.ylabel(f"PC2 ({var_pc2*100:.1f}% var)")
        plt.xlim(-0.9, 0.9)
        plt.ylim(-0.9, 0.9)
        plt.xticks(np.arange(-0.9, 0.91, 0.1))
        plt.yticks(np.arange(-0.9, 0.91, 0.1))
        plt.axhline(0, color='black', linewidth=1.5)
        plt.axvline(0, color='black', linewidth=1.5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        safe_label = label_suffix.replace(" ", "_").replace("/", "-")

        # --- Save figure ---
        png_path = os.path.join(OUTPUT_FOLDER, f"PCA_{lang}_{safe_label}.png")
        plt.savefig(png_path, dpi=600)
        plt.close()

        # --- Save textual data ---
        txt_path = os.path.join(OUTPUT_FOLDER, f"PCA_{lang}_{safe_label}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"PCA RESULTS ({lang}) {label_suffix}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Explained variance:\n")
            for i, v in enumerate(pca.explained_variance_ratio_):
                f.write(f"  PC{i+1}: {v*100:.2f}%\n")
            f.write(f"\nTotal variance explained (first {n_components} PCs): {total_var:.4f}\n\n")

            f.write("--- DOCUMENT COORDINATES ---\n")
            f.write("Label\tPC1\tPC2\n")
            for label in doc_coords.index:
                f.write(f"{label}\t{doc_coords.loc[label, 'PC1']:.5f}\t{doc_coords.loc[label, 'PC2']:.5f}\n")

            f.write("\n--- TOP WORD LOADINGS ---\n")
            f.write("Word\tPC1\tPC2\n")
            for word in top_words_idx:
                f.write(f"{word}\t{loadings.loc[word, 'PC1']:.5f}\t{loadings.loc[word, 'PC2']:.5f}\n")

            f.write("\n--- HIGHLIGHT WORDS (green) ---\n")
            f.write("Word\tPC1\tPC2\n")
            for hw in highlight_words:
                if hw in matrix_df.index:
                    idx = matrix_df.index.get_loc(hw)
                    v = np.zeros(matrix_df.shape[0])
                    v[idx] = 1
                    coord = pca.transform(v.reshape(1, -1))
                    f.write(f"{hw}\t{coord[0,0]:.5f}\t{coord[0,1]:.5f}\n")

        print(f"Saved PCA plot and data to:\n  {png_path}\n  {txt_path}")

    # --- 1. Global PCA by canton ---
    df_canton = build_matrix(kwic_rows, lang, by="canton")
    _pca_and_plot(df_canton, label_suffix="by canton", by="canton")

    # --- 2. Global PCA by year ---
    df_year = build_matrix(kwic_rows, lang, by="year")
    _pca_and_plot(df_year, label_suffix="by year", by="year")

    # --- 3. Per-canton PCA by year ---
    canton_groups = defaultdict(list)
    for w in kwic_rows:
        canton_groups[w["canton"]].append(w)

    for canton, rows in canton_groups.items():
        df_canton_year = build_matrix(rows, lang, by="year")
        if df_canton_year.shape[1] >= 2:
            _pca_and_plot(df_canton_year, label_suffix=f"by year – {canton}", by="year")
        else:
            print(f"Skipping {canton}: not enough yearly data for PCA")




def run_canton_year_plot(kwic_rows, lang):
    """Visualize trend of cantons across years based on top lemma occurrence."""
    # Build canton x year frequency
    df = pd.DataFrame(kwic_rows)
    canton_year = df.groupby(["canton", "year"]).size().unstack(fill_value=0)
    canton_year_norm = canton_year.div(canton_year.sum(axis=1), axis=0)

    canton_year_norm.T.plot(kind='line', figsize=(9, 6), marker='o')
    plt.title(f"{lang}: Canton trends over years (normalized counts)")
    plt.xlabel("Year")
    plt.ylabel("Normalized occurrence")
    plt.legend(title="Canton")
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_FOLDER, f"Canton{lang}TrendOverYears.png")
    plt.savefig(png_path, dpi=600)
    #plt.show()

# ----------------------
# PART 6: TABLE COUNTS EXPORT
# ----------------------

def count_wordlist_occurrences(dataset, lang):
    """
    Count occurrences of every lemma appearing in WORDLIST_FILES for a given language.
    Supports multiword lemmas: all words must occur within WINDOW_MULTI words.
    Returns a dictionary {lemma: count}, including 0 for unseen lemmas.
    """
    lemma_key = f"{lang.lower()}lemma"
    all_wordlists = load_wordlist(lang)
    target_lemmas = set(word for words in all_wordlists.values() for word in words)

    counter = Counter()

    # Preprocess dataset into a list of lemmas for easier multiword search
    dataset_lemmas = [w[lemma_key].lower() for w in dataset]

    for lemma in target_lemmas:
        words = lemma.split()  # Split multiword lemmas into components
        if len(words) == 1:
            # Single-word lemma: simple count
            counter[lemma] = dataset_lemmas.count(lemma)
        else:
            # Multiword lemma: check if all words occur within WINDOW_MULTI in sequence
            count = 0
            for i, w in enumerate(dataset_lemmas):
                if w == words[0]:
                    # Look ahead WINDOW_MULTI+len(words)-1 positions
                    end_idx = min(i + len(words) + WINDOW_MULTI - 1, len(dataset_lemmas))
                    window = dataset_lemmas[i:end_idx]
                    if all(word in window for word in words[1:]):
                        count += 1
            counter[lemma] = count

    # Add unseen lemmas explicitly with count = 0
    for lemma in target_lemmas:
        counter.setdefault(lemma, 0)

    return dict(counter)


def export_wordlist_counts(dataset, lang):
    """
    Export the counts of all wordlist lemmas for a language into a .txt file.
    """
    counts = count_wordlist_occurrences(dataset, lang)
    txt_path = os.path.join(OUTPUT_FOLDER, f"{lang}_wordlist_counts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for lemma, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{lemma}\t{count}\n")
    print(f"✅ Exported {len(counts)} word counts for {lang} to {txt_path}")

# ----------------------
# MAIN EXECUTION
# ----------------------
def main():
    for lang in ["FR", "DE"]:
        # PART 1
        print("Loading dataset")
        dataset = load_dataset(lang)

        # PART 2
        print("Computing Stat on POS")
        pos_statistics(dataset, lang)

        # PART 2.5
        print("Counting all target word occurrences for table export")
        export_wordlist_counts(dataset, lang)

        # PART 3
        print("filter dataset by keyword")
        wordlists = load_wordlist(lang)
        filtered, lemma_counts = filter_dataset(dataset, lang, wordlists)

        # Export filtered dataset
        filtered_csv = os.path.join(OUTPUT_FOLDER, f"{lang}_filtered_dataset.csv")
        pd.DataFrame(filtered).to_csv(filtered_csv, index=False, encoding="utf-8")

        # Export lemma occurrence counts
        lemma_counts_csv = os.path.join(OUTPUT_FOLDER, f"{lang}_filtered_lemma_counts.csv")
        pd.DataFrame(lemma_counts).sort_values(["list", "count"], ascending=[True, False]).to_csv(lemma_counts_csv, index=False, encoding="utf-8")


        # PART 4
        print("computing KWIC")
        kwic_rows = generate_kwic(filtered,dataset, lang)
        kwic_csv = os.path.join(OUTPUT_FOLDER, f"{lang}_kwic.csv")
        pd.DataFrame(kwic_rows).to_csv(kwic_csv, index=False, encoding="utf-8")

        # PART 5
        print("running AFC")
        run_pca(kwic_rows, lang=lang)

        # PART 6
        #print("computing over time trend")
        #run_canton_year_plot(kwic_rows, lang=lang)

if __name__ == "__main__":
    main()
