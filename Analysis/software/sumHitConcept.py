import pandas as pd
import os

# ----------------------
# CONFIGURATION
# ----------------------

DE_FILE = os.path.join("result", "DE_filtered_lemma_counts.csv")
FR_FILE = os.path.join("result", "FR_filtered_lemma_counts.csv")

# ----------------------
# KEYWORDS
# ----------------------
# Economic / Performance vs Behavioral / Experiential
economic_keywords = ["Kosten", "coût", "Leistung", "efficacité", "optimisation", "optimal"]
behavioral_keywords = ["Erfahrung", "experience", "perception", "Wahrnehmung", "confort", "fluidité"]

# Accessibility vs Motility
accessibility_keywords = ["Erreichbarkeit", "accessibilité", "Zugänglichkeit"]
motility_keywords = ["Mobilität", "motilité", "Bewegung", "Mobilitätsstrategie", "ressource mobilité"]

# Time: Objective vs Perceived
objective_time_keywords = [
    "Zeit", "temps", "Dauer", "durée", "Stunde", "minute", "heures", "vitesse",
    "Kilometer", "kms", "Ponctualität", "fiabilité", "Leistung", "Leistung"
]
perceived_time_keywords = [
    "Erfahrung", "experience", "perception", "Wahrnehmung", "confort", "fluidité",
    "Erlebnis", "satisfaction", "qualité", "qualité de vie", "ressenti"
]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def count_hits(df, keywords, lemma_col="lemma", count_col="count"):
    """
    Count occurrences of keywords using precomputed counts in CSV.
    """
    df_filtered = df[df[lemma_col].isin(keywords)]
    return df_filtered[count_col].sum()

# ----------------------
# MAIN
# ----------------------
def main():
    # Read German and French lemma counts CSVs
    df_de = pd.read_csv(DE_FILE)
    df_fr = pd.read_csv(FR_FILE)

    # --- Behavioral vs Economic ---
    econ_de = count_hits(df_de, economic_keywords)
    econ_fr = count_hits(df_fr, economic_keywords)
    beh_de = count_hits(df_de, behavioral_keywords)
    beh_fr = count_hits(df_fr, behavioral_keywords)

    print(f"""
% Behavioral vs Economic
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
Category & Hits DE & Hits FR \\\\
\\hline
Economic / Performance Rationales & {econ_de} & {econ_fr} \\\\
Behavioral / Experiential Rationales & {beh_de} & {beh_fr} \\\\
\\hline
\\end{{tabular}}
\\caption{{Behavioral vs Economic framing in German and French planning documents.}}
\\end{{table}}
""")

    # --- Accessibility vs Motility ---
    acc_de = count_hits(df_de, accessibility_keywords)
    acc_fr = count_hits(df_fr, accessibility_keywords)
    mot_de = count_hits(df_de, motility_keywords)
    mot_fr = count_hits(df_fr, motility_keywords)

    print(f"""
% Accessibility vs Motility
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
Category & Hits DE & Hits FR \\\\
\\hline
Accessibility (Classical) & {acc_de} & {acc_fr} \\\\
Motility (Potential / Capability) & {mot_de} & {mot_fr} \\\\
\\hline
\\end{{tabular}}
\\caption{{Accessibility vs Motility in German and French planning documents.}}
\\end{{table}}
""")

    # --- Objective vs Perceived Time ---
    obj_de = count_hits(df_de, objective_time_keywords)
    obj_fr = count_hits(df_fr, objective_time_keywords)
    perc_de = count_hits(df_de, perceived_time_keywords)
    perc_fr = count_hits(df_fr, perceived_time_keywords)

    print(f"""
% Objective vs Perceived Time
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
Category & Hits DE & Hits FR \\\\
\\hline
Objective Time & {obj_de} & {obj_fr} \\\\
Perceived Time & {perc_de} & {perc_fr} \\\\
\\hline
\\end{{tabular}}
\\caption{{Objective vs Perceived Time in German and French planning documents.}}
\\end{{table}}
""")

if __name__ == "__main__":
    main()
