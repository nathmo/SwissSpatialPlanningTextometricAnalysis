
# Analyse textométrique des plans d'aménagement du territoire suisse

This repository contains all materials for a reproducible textometric analysis of Swiss spatial planning documents. It includes the code, dataset, and report so that anyone can explore, replicate, or extend the research.

## Contents

* **PDF report:** `Analyse_textométrique_des_plan_d'aménagement_du_territoire_suisse.pdf`
  This is the full research document detailing the questions, methodology, results, and conclusions.
  **Abstract:** The study investigates how accessibility, motility, and time are represented in Swiss spatial planning discourse. Accessibility dominates the discourse, while capability-based mobility (motility) and subjective time are largely absent, revealing a bias toward technical and operational framings. The project provides an open dataset and analysis pipeline to facilitate further research.

* **Dataset:** Raw and processed text data extracted from cantonal planning PDFs.

* **Analysis scripts:** Main script located at:

  ```
  Analysis/software/analysis.py
  ```

  This script processes the text data, computes counts for mobility and time concepts, and generates matrices and visualizations for analysis.

* **Requirements:** `requirements.txt` contains all Python dependencies.

## Quick start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the analysis:

   ```
   python Analysis/software/analysis.py
   ```

The analysis pipeline reproduces the results presented in the PDF report. It also allows exploration of combined French and German datasets, subcorpora filtered by concept relevance, and concept-level matrices for Accessibility, Motility, and Time (both objective and subjective).

## Open Science Notes

* The report and datasets are released under a **CC-BY license** to encourage reuse and replication.
* The script are release under a **MIT License** for easy reuse.
