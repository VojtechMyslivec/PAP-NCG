#!/bin/bash

# metoda vystupu z programu uzly | matice
metoda="uzly"

data="../data/male-grafy"
vystupy="./vystupy"
rozdily="$vystupy/diff"
dijkstra="./dijkstra"
floyd="./floyd-warshall"

# kontrola spustitelnych souboru
[[ -r "$floyd" && -x "$floyd" && -r "$dijkstra" && -x "$dijkstra" ]] || {
    echo "Soubory '$floyd' a '$dijkstra' musi byt spustitelne programy." >&2
    exit 1
}

# kontrola existence vstupnich dat
ls "$data/"*.txt >/dev/null 2>&1 || {
    echo "Neexistuji vstupni data v adresari '$data'!" >&2
    exit 1
}

# Pokud neexistuji adresare, vytvori je
mkdir -p "$rozdily"


# Buildovani programu pres make
# make > /dev/null
# [[ $? -ne 0 ]] && {
#    echo "Chyba (make)! Nelze provest prikaz make, chybi Makefile!" >&2
#    exit 1
# }

# Spousteni pro ruzne vstupni soubory
for file in "$data"/*.txt ; do
   filename=${file##*/}
   vystupD="${vystupy}/out_${metoda}_dijkstra_${filename}"
   vystupF="${vystupy}/out_${metoda}_floyd_${filename}"
   vystupDiff="${rozdily}/diff_${metoda}_${filename}"

   echo "Zpracovavam soubor '$filename'"
   "$dijkstra" -t 6 -f "${file}" > "${vystupD}" 2> /dev/null
   "$floyd"    -t 6 -f "${file}" > "${vystupF}" 2> /dev/null
   diff "${vystupD}" "${vystupF}" > "${vystupDiff}"
done

# Uklid souboru
# make clean > /dev/null
# [[ $? -ne 0 ]] && {
#    echo "Chyba (make clean)! Nepodarilo se smazat vsechny vystupni soubory" >&2
#    exit 2
# }

exit 0

