# Analyse textométrique des plan d'aménagement du territoire suisse

This repository contains the LaTeX document of my analysis, the scripts used to process the dataset, and the dataset itself.

My research questions are the followings :

To what extent do Swiss territorial / spatial planning documents acknowledge the distinction
between perceived time vs measured/travel-time objectively modelled, in relation to modal
choice?

How strongly do Swiss planning documents incorporate human / behavioral / motility-based
rationales, as opposed to purely economic / efficiency rationales, in shaping transport and spatial
policy?

# Methodology

I manually downloaded the PDF on each canton's website.

The raw text was extracted from the pdf using a bash one liner

```
for f in *.pdf *.PDF; do [ -f "$f" ] && pdftotext "$f" "${f%.*}.txt"; done
```

I then imported the .txt into TXM : https://txm.gitpages.huma-num.fr/textometrie/

Heiden Serge. (2010). The TXM Platform: Building Open-Source Textual Analysis Software Compatible with the TEI Encoding Scheme. In 24th Pacific Asia Conference on Language, Information and Computation (pp. 389–398). Sendai, Japon. Retrieved from http://halshs.archives-ouvertes.fr/docs/00/54/97/64/PDF/paclic24_sheiden.pdf

TXM use the treetagger module to tag each word in the corpus. (https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)

I used the French and German module for the French and German dataset.

From there I made a custom python script to process the data.