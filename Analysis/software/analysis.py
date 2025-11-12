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
import csv
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

WHITELIST_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "ContextMobiliteWhitelist.csv"))
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

WINDOW_MULTI = 3  # max words allowed between multi-word reference words

# smoothing / segmentation params (tweakable)
SMOOTH_SIGMA = 50.0   # gaussian kernel std (in tokens) for smoothing density
THRESH_QUANTILE = 0.9  # retain regions above this percentile of density Quantiles must be in the range [0, 1]

# ----------------------
# PART 1: LOAD XML DATA
# ----------------------
def parse_xml_file(file_path, lang):
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
            w_id = w.get("id")
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
    unk_count = sum(1 for w in dataset if w[f"{lang.lower()}pos"] == "UNK")
    if unk_count > 0:
        print(f"⚠️ {unk_count} words with UNK {lang} POS")
    return dataset

# ----------------------
# PART 2: POS STATISTICS (unchanged)
# ----------------------
def pos_statistics(dataset, lang):
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

        csv_path = os.path.join(OUTPUT_FOLDER, f"{lang}_{pos}_lemmas.csv")
        pd.DataFrame(lemma_counter.items(), columns=["lemma", "count"])\
          .sort_values("count", ascending=False)\
          .to_csv(csv_path, index=False, encoding="utf-8")

        summary_stats.append({
            "POS": pos,
            "total_tokens": total_tokens,
            "unique_lemmas": unique_lemmas
        })

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
# PART 3: WORDLISTS & MULTIWORD MATCH
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
    ref_tokens = ref_expr.split()
    matches = []
    i = 0
    while i <= len(lemmas) - len(ref_tokens):
        idx_list = [i]
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

# ----------------------
# NEW: load whitelist CSV -> lemma -> score
# ----------------------
def load_context_whitelist(path=WHITELIST_FILE):
    mapping = {}
    if not os.path.exists(path):
        print(f"⚠️ Whitelist CSV not found at {path}. All scores will be zero.")
        return mapping
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'lemma' not in reader.fieldnames or 'score' not in reader.fieldnames:
            print(f"Found Header : {reader.fieldnames}")
            raise ValueError("ContextMobiliteWhitelist.csv must have 'lemma' and 'score' columns")
        for r in reader:
            try:
                lemma = r['lemma'].strip().lower()
                score = float(r['score'])
                mapping[lemma] = score
            except Exception as e:
                # skip malformed lines but warn
                print(f"⚠️ Skipping whitelist line {r}: {e}")
    return mapping

# ----------------------
# PART 4: token-level scoring, smoothing, segmentation
# ----------------------
def compute_token_scores(dataset, mapping, lang):
    lemma_key = f"{lang.lower()}lemma"
    # Build list of token lemmas aligned with dataset
    lemmas = [w[lemma_key].lower() for w in dataset]
    scores = [mapping.get(l, 0.0) for l in lemmas]
    return np.array(scores, dtype=float), lemmas

def gaussian_kernel(sigma, radius_factor=4):
    # radius_factor*sigma on each side
    radius = max(1, int(radius_factor * sigma))
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    return kernel

def smooth_scores(scores, sigma=SMOOTH_SIGMA):
    if sigma <= 0 or len(scores) == 0:
        return scores
    kernel = gaussian_kernel(sigma)
    smoothed = np.convolve(scores, kernel, mode='same')
    return smoothed

def plot_density_histogram(density, lang):
    plt.figure(figsize=(8,4))
    plt.hist(density, bins=100)
    plt.title(f"{lang} mobility concept density histogram")
    plt.xlabel("Density (smoothed score)")
    plt.ylabel("Frequency")
    out = os.path.join(OUTPUT_FOLDER, f"{lang}_mobility_density_hist.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved density histogram to {out}")

def segment_by_threshold(density, quantile=THRESH_QUANTILE, min_segment_len=5):
    thresh = float(np.quantile(density, quantile))
    mask = density >= thresh
    segments = []
    if not mask.any():
        return segments, thresh
    # get runs of True in mask
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_segment_len:
                segments.append((start, i-1))
            start = None
    if start is not None and len(mask) - start >= min_segment_len:
        segments.append((start, len(mask)-1))
    return segments, thresh

# helper to extract subcorpus
def extract_subcorpus(dataset, segments):
    # dataset is list of token dicts aligned with segments indices
    kept = []
    for s,e in segments:
        kept.extend(dataset[s:e+1])
    return kept

# ----------------------
# PART 5: filtering + counts per document for each wordlist (on subcorpus)
# ----------------------
def count_wordlist_per_document(subcorpus, lang, wordlist_entries):
    """
    wordlist_entries: list of lemma expressions (could be multiword)
    Returns DataFrame rows=lemmas (in the given order), cols=documents, with counts
    """
    lemma_key = f"{lang.lower()}lemma"
    # Build per-document list of lemmas (and original indexes)
    docs = defaultdict(list)  # doc -> list of (index_in_subcorpus, lemma)
    for idx, w in enumerate(subcorpus):
        docs[w['document']].append((idx, w[lemma_key].lower()))

    # We'll produce a DataFrame with rows = wordlist_entries (in given order)
    all_docs = sorted(docs.keys())
    df = pd.DataFrame(0, index=wordlist_entries, columns=all_docs, dtype=int)

    # For efficiency, prebuild per-document lemma list plain
    for doc in all_docs:
        indices, lemmas = zip(*docs[doc]) if docs[doc] else ([], [])
        lemmas = list(lemmas)
        # for each entry (could be multiword)
        for entry in wordlist_entries:
            if " " not in entry:
                cnt = lemmas.count(entry)
                df.at[entry, doc] = cnt
            else:
                # multiword: find matches using the same algorithm
                matches = match_multiword(lemmas, entry)
                df.at[entry, doc] = len(matches)
    return df

# ----------------------
# PART 6: helper to build concept matrices and merge bilingual sets
# ----------------------
def build_concept_matrices(subcorpus, lang, wordlists):
    """
    wordlists: dict with keys as filenames -> list of lemmas
    we expect keys containing 'Access' or 'Motil' or 'TempsChrono' or 'TempsVecue'
    Returns a dict:
      {
        'time': dataframe (rows words, cols documents)  # time = chrono + vecue
        'mobility': dataframe (rows words, cols documents) # mobility = access + motil
        'by_list': {list_key: df}
      }
    """
    # detect lists
    time_keys = [k for k in wordlists.keys() if 'Temps' in k or 'temps' in k or 'Chrono' in k or 'chrono' in k or 'Temps' in k]
    vecue_keys = [k for k in wordlists.keys() if 'Vecue' in k or 'vecue' in k]
    # but simpler: use exact base filenames if provided in WORDLIST_FILES; we'll assume four lists per language as before:
    # Accessibilité -> index 0, Motilité -> index 1, TempsChrono -> index 2, TempsVecue -> index 3 (this was your original order)
    list_order = list(wordlists.keys())
    # flatten groups:
    access = []
    motil = []
    chrono = []
    vecue = []
    for k in list_order:
        lname = k.lower()
        if 'access' in lname or 'accessibilité' in lname or 'accessibilit' in lname:
            access.extend(wordlists[k])
        elif 'motil' in lname:
            motil.extend(wordlists[k])
        elif 'chrono' in lname or 'tempschrono' in lname:
            chrono.extend(wordlists[k])
        elif 'vecue' in lname or 'tempsvecue' in lname:
            vecue.extend(wordlists[k])
        else:
            # fallback add to appropriate buckets by presence of key words
            if 'temps' in lname:
                chrono.extend(wordlists[k])
            else:
                # as fallback append to access
                access.extend(wordlists[k])

    # prepare lists lowercased and keep order and uniqueness
    def norm_list(lst):
        seen = set()
        out = []
        for w in lst:
            ww = w.lower()
            if ww not in seen:
                seen.add(ww)
                out.append(ww)
        return out

    access = norm_list(access)
    motil = norm_list(motil)
    chrono = norm_list(chrono)
    vecue = norm_list(vecue)

    time_list = chrono + vecue
    mob_list = access + motil

    results = {'by_list': {}}
    # compute per-list DF
    for key, entries in [('Accessibilité', access), ('Motilité', motil), ('TempsChrono', chrono), ('TempsVecue', vecue),
                         ('Time', time_list), ('Mobility', mob_list)]:
        if not entries:
            df = pd.DataFrame()
        else:
            df = count_wordlist_per_document(subcorpus, lang, entries)
        results['by_list'][key] = df
        if key == 'Time':
            results['time'] = df
        if key == 'Mobility':
            results['mobility'] = df
    return results

def save_matrix_csv(df, name):
    if df is None or df.empty:
        print(f"Matrix {name} is empty; skipping CSV save.")
        return
    path = os.path.join(OUTPUT_FOLDER, f"{name}.csv")
    df.to_csv(path, encoding='utf-8')
    print(f"Saved matrix CSV: {path}")

# ----------------------
# PART 7: PCA / AFC on the six matrices
# ----------------------
def pca_and_save(matrix_df, label_suffix):
    if matrix_df is None or matrix_df.empty:
        print(f"Skipping PCA {label_suffix}: empty matrix")
        return
    # rows = words, cols = documents
    # to be consistent with previous PCA code: transpose and run PCA on columns
    try:
        matrix_df = matrix_df.astype(float)
        safe_label = label_suffix.replace(" ", "_").replace("/", "-")
        # save CSV
        save_matrix_csv(matrix_df, f"MATRIX_{safe_label}")

        pca = PCA(n_components=min(10, matrix_df.shape[0], matrix_df.shape[1]))
        coords = pca.fit_transform(matrix_df.T)

        doc_coords = pd.DataFrame(coords, index=matrix_df.columns,
                                  columns=[f"PC{i+1}" for i in range(coords.shape[1])])
        loadings = pd.DataFrame(pca.components_.T, index=matrix_df.index,
                                columns=[f"PC{i+1}" for i in range(coords.shape[1])])

        var_pc1 = pca.explained_variance_ratio_[0] if pca.explained_variance_ratio_.size > 0 else 0.0
        var_pc2 = pca.explained_variance_ratio_[1] if pca.explained_variance_ratio_.size > 1 else 0.0

        # --- NORMALIZE DOCS AND WORDS TO [-1, 1] ---
        def normalize_axis(df):
            df_norm = df.copy()
            for axis in ["PC1", "PC2"]:
                min_val = df[axis].min()
                max_val = df[axis].max()
                if max_val > min_val:
                    df_norm[axis] = 2 * (df[axis] - min_val) / (max_val - min_val) - 1
                else:
                    df_norm[axis] = 0.0
            return df_norm

        doc_coords_norm = normalize_axis(doc_coords)
        loadings_norm = normalize_axis(loadings)

        # --- PLOTTING ---
        plt.figure(figsize=(9, 6))
        # draw black lines at 0.0 for reference
        plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
        plt.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)

        # plot documents
        plt.scatter(doc_coords_norm["PC1"], doc_coords_norm["PC2"], alpha=0.6, color="steelblue")
        texts = []
        for label in doc_coords_norm.index:
            # Extract the document name (remove language or prefixes like "FR_", "DE_", etc.)
            short_label = os.path.basename(str(label))
            # Optional: if your labels are like "FR_doc1", take only part after underscore
            if "_" in short_label:
                short_label = short_label.split("_")[-1]

            x, y = doc_coords_norm.loc[label, "PC1"], doc_coords_norm.loc[label, "PC2"]
            texts.append(plt.text(x, y, short_label, fontsize=8, alpha=0.7, color="blue"))


        # plot top words
        word_counts = matrix_df.sum(axis=1)
        top_words_idx = word_counts.sort_values(ascending=False).head(
            20).index if not word_counts.empty else matrix_df.index
        for word in top_words_idx:
            x, y = loadings_norm.loc[word, "PC1"], loadings_norm.loc[word, "PC2"]
            plt.scatter(x, y, color="red", s=20)
            texts.append(plt.text(x + 0.02, y + 0.02, str(word), fontsize=7, color="red", alpha=0.8))

        adjust_text(texts, only_move={'points': 'y', 'texts': 'xy'},
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))
        plt.title(f"PCA {label_suffix}")
        plt.xlabel(f"PC1 ({var_pc1 * 100:.1f}% var)")
        plt.ylabel(f"PC2 ({var_pc2 * 100:.1f}% var)")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        png_path = os.path.join(OUTPUT_FOLDER, f"PCA_{safe_label}.png")
        plt.savefig(png_path, dpi=600)
        plt.close()

        txt_path = os.path.join(OUTPUT_FOLDER, f"PCA_{safe_label}.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(f"PCA RESULTS {label_suffix}\n")
            f.write("="*60 + "\n\n")
            for i, v in enumerate(pca.explained_variance_ratio_):
                f.write(f"  PC{i+1}: {v*100:.2f}%\n")
            f.write("\n--- DOCUMENT COORDINATES ---\n")
            f.write("Label\tPC1\tPC2\n")
            for label in doc_coords.index:
                f.write(f"{label}\t{doc_coords.loc[label, 'PC1']:.5f}\t{doc_coords.loc[label, 'PC2']:.5f}\n")
            f.write("\n--- TOP WORD LOADINGS ---\n")
            f.write("Word\tPC1\tPC2\n")
            for word in top_words_idx:
                f.write(f"{word}\t{loadings.loc[word, 'PC1']:.5f}\t{loadings.loc[word, 'PC2']:.5f}\n")
        print(f"Saved PCA outputs: {png_path}, {txt_path}")
    except Exception as e:
        print(f"Error in PCA {label_suffix}: {e}")

# ----------------------
# MAIN EXECUTION (implements pipeline you asked for)
# ----------------------
def main():
    # 0) load whitelist mapping
    whitelist = load_context_whitelist(WHITELIST_FILE)

    # 1) load datasets and compute POS stats
    datasets = {}
    for lang in ["FR","DE"]:
        print(f"Loading dataset {lang}")
        datasets[lang] = load_dataset(lang)
        print(f"Computing POS stats for {lang}")
        pos_statistics(datasets[lang], lang)

    # 2) For each language: compute token-level scores, smooth, plot histogram, segment
    subcorpora = {}
    for lang in ["FR","DE"]:
        print(f"Scoring tokens for {lang}")
        scores, lemmas = compute_token_scores(datasets[lang], whitelist, lang)
        print("Smoothing scores...")
        smooth = smooth_scores(scores, sigma=SMOOTH_SIGMA)
        # plot histogram
        plot_density_histogram(smooth, lang)
        # segment by threshold
        segments, thresh = segment_by_threshold(smooth, quantile=THRESH_QUANTILE)
        print(f"{lang}: threshold (quantile {THRESH_QUANTILE}) = {thresh:.6f}, segments found: {len(segments)}")
        # optionally save segments summary
        seg_path = os.path.join(OUTPUT_FOLDER, f"{lang}_segments.txt")
        with open(seg_path, "w", encoding='utf-8') as f:
            f.write(f"threshold\t{thresh}\n")
            for s,e in segments:
                f.write(f"{s}\t{e}\n")
        # build subcorpus = tokens in those segments
        sub = extract_subcorpus(datasets[lang], segments)
        subcorpora[lang] = sub
        print(f"{lang}: kept {len(sub)} tokens out of {len(datasets[lang])}")
        # --- Export readable segments (actual text) ---
        seg_txt_path = os.path.join(OUTPUT_FOLDER, f"{lang}_segments_text.txt")
        with open(seg_txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Segmented text for {lang}\n")
            f.write(f"# Threshold (quantile {THRESH_QUANTILE}) = {thresh}\n\n")

            for i, (s, e) in enumerate(segments, start=1):
                # join lemmas into readable text
                segment_words = " ".join(lemmas[s:e+1])

                # light punctuation cleanup
                segment_words = segment_words.replace(" ,", ",").replace(" .", ".").replace(" '", "'")

                f.write(f"[SEGMENT {i}] ({s}–{e})\n")
                f.write(segment_words.strip() + "\n\n")

        print(f"Saved readable text segments for {lang} → {seg_txt_path}")



    # 3) For each language compute detailed counts per document for each list (on subcorpus)
    matrices = {}  # matrices[lang][listkey] = DataFrame
    for lang in ["FR","DE"]:
        print(f"Building concept matrices for {lang}")
        wordlists = load_wordlist(lang)
        res = build_concept_matrices(subcorpora[lang], lang, wordlists)
        matrices[lang] = res['by_list']
        # save CSVs for each list
        for k, df in res['by_list'].items():
            save_matrix_csv(df, f"{lang}_{k}")
        # also save Time and Mobility matrices
        save_matrix_csv(res.get('time', pd.DataFrame()), f"{lang}_Time")
        save_matrix_csv(res.get('mobility', pd.DataFrame()), f"{lang}_Mobility")

    # 4) Merge FR+DE for Time and Mobility (pairing lemmas by index)
    def merge_bilingual(fr_df, de_df, fr_list, de_list, label_prefix):
        # fr_df rows = fr_list, cols = fr_docs ; de_df rows = de_list, cols = de_docs
        if (fr_df is None or fr_df.empty) and (de_df is None or de_df.empty):
            return pd.DataFrame()
        # determine min length for pairing
        n = min(len(fr_list), len(de_list))
        if n == 0:
            print(f"⚠️ Cannot merge {label_prefix}: one list empty")
            return pd.DataFrame()
        merged_labels = [f"{fr_list[i]} | {de_list[i]}" for i in range(n)]
        fr_docs = list(fr_df.columns) if (fr_df is not None and not fr_df.empty) else []
        de_docs = list(de_df.columns) if (de_df is not None and not de_df.empty) else []
        cols = [f"{d}" for d in fr_docs] + [f"{d}" for d in de_docs]
        merged = pd.DataFrame(0, index=merged_labels, columns=cols, dtype=int)
        # fill FR part
        for i in range(n):
            frw = fr_list[i]
            for d in fr_docs:
                if frw in fr_df.index and d in fr_df.columns:
                    merged.at[merged_labels[i], f"{d}"] = int(fr_df.at[frw, d])
        # fill DE part
        for i in range(n):
            dew = de_list[i]
            for d in de_docs:
                if dew in de_df.index and d in de_df.columns:
                    merged.at[merged_labels[i], f"{d}"] = int(de_df.at[dew, d])
        return merged

    # Prepare lists for pairing using the same logic as build_concept_matrices
    fr_wordlists = load_wordlist("FR")
    de_wordlists = load_wordlist("DE")

    # Normalize and get ordered lists from wordlists (same normalization used earlier)
    def flatten_and_norm(wordlists):
        out = []
        for k in wordlists.keys():
            for w in wordlists[k]:
                ww = w.lower()
                if ww not in out:
                    out.append(ww)
        return out

    fr_all = flatten_and_norm(fr_wordlists)
    de_all = flatten_and_norm(de_wordlists)
    # Now construct time lists (chrono + vecue) and mobility lists (access + motil) in same manner as before
    def extract_concept_lists(wordlists):
        access = []
        motil = []
        chrono = []
        vecue = []
        for k in wordlists.keys():
            lname = k.lower()
            items = [w.lower() for w in wordlists[k]]
            if 'access' in lname or 'accessibilit' in lname:
                access.extend(items)
            elif 'motil' in lname:
                motil.extend(items)
            elif 'chrono' in lname or 'tempschrono' in lname:
                chrono.extend(items)
            elif 'vecue' in lname or 'tempsvecue' in lname:
                vecue.extend(items)
            else:
                if 'temps' in lname:
                    chrono.extend(items)
                else:
                    access.extend(items)
        def unique_ordered(lst):
            seen=set(); out=[]
            for w in lst:
                if w not in seen:
                    seen.add(w); out.append(w)
            return out
        return unique_ordered(access), unique_ordered(motil), unique_ordered(chrono), unique_ordered(vecue)

    fr_access, fr_motil, fr_chrono, fr_vecue = extract_concept_lists(fr_wordlists)
    de_access, de_motil, de_chrono, de_vecue = extract_concept_lists(de_wordlists)

    fr_time_list = fr_chrono + fr_vecue
    de_time_list = de_chrono + de_vecue
    fr_mob_list = fr_access + fr_motil
    de_mob_list = de_access + de_motil

    # get the per-list DataFrames computed earlier
    fr_time_df = matrices['FR'].get('Time', pd.DataFrame())
    fr_mob_df = matrices['FR'].get('Mobility', pd.DataFrame())
    de_time_df = matrices['DE'].get('Time', pd.DataFrame())
    de_mob_df = matrices['DE'].get('Mobility', pd.DataFrame())

    merged_time = merge_bilingual(fr_time_df, de_time_df, fr_time_list, de_time_list, "Time")
    merged_mob = merge_bilingual(fr_mob_df, de_mob_df, fr_mob_list, de_mob_list, "Mobility")

    save_matrix_csv(merged_time, "MERGED_Time")
    save_matrix_csv(merged_mob, "MERGED_Mobility")

    # 5) Run PCA/AFC on the six matrices:
    # FR Time, FR Mobility, DE Time, DE Mobility, MERGED Time, MERGED Mobility
    pca_and_save(fr_time_df, "FR_Time")
    pca_and_save(fr_mob_df, "FR_Mobility")
    pca_and_save(de_time_df, "DE_Time")
    pca_and_save(de_mob_df, "DE_Mobility")
    pca_and_save(merged_time, "MERGED_Time")
    pca_and_save(merged_mob, "MERGED_Mobility")

    print("Pipeline finished. Check the result/ folder for CSVs, PCA plots and text outputs.")

if __name__ == "__main__":
    main()
