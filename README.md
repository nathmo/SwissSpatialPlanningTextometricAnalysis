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

First we compute some stat on the corpus 
then we filter it by scoring word positively or negatively if they are close the mobility concept
by matching them using ContextMobilityWhiltelist.csv (word, score) with score being positive or negative if its related or not to the concept of mobility)
we then plot the concept density histogram.
then we cluster/apply a density threshold to only keep the relevant part of the dataset that are related to mobility and exclude the one that belong to other topics like energy.
then on that subcorpus we fill a csv/table with the count per word for each lemmes (Accessibilité / Motilité/TempsChrono/TempsVecue) and for each document. we now have a matrix of vector per document that tell us how many hit we have for each word.
the goal being to give the count per detailled concept (timeChrono, TempsVecue) but do the AFC on the whole concept (Time) to see if there is a difference between documents.
since the word are matched between german and french we can even make a new CSV that merge the column of the french + german CSV for the time + mobility concept.
then we can compute the AFC on the french Time csv + french mobilité csv + german time csv + german mobility csv + the merge time csv and the merged mobility csv.


