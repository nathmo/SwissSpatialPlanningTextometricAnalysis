echo "id,canton,language,year" > metadata.csv
for f in *.txt; do
    canton=$(echo "$f" | cut -d'_' -f2)
    lang="${f:0:2}"
    year=$(echo "$f" | cut -d'_' -f3 | cut -d'.' -f1)
    echo "$f,$canton,$lang,$year" >> metadata.csv
done
