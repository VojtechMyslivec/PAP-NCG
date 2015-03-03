#!/bin/bash

# metoda vystupu z programu uzly | matice
metoda="uzly"

data="../data"
vystupy="./vystupy"
rozdily="$vystupy/diff"
dijkstra="./dijkstra"
floyd="./floyd-warshall"

# Pokud neexistuji adresare, vytvori je
[[ ! -d "$vystupy" ]] && {
   mkdir "$vystupy"
}
[[ ! -d "$rozdily" ]] && {
   mkdir "$rozdily"
}


# Buildovani programu pres make
make
[[ $? -ne 0 ]] && {
   echo "Chyba (make)! Nelze provest prikaz make, chybi Makefile!" >&2
   exit 1
}

# Spousteni pro ruzne vstupni soubory
for file in "$data"/* ; do
   file=`basename "$file"`
   vystupD="${vystupy}/out_${metoda}_dijkstra_${file}"
   vystupF="${vystupy}/out_${metoda}_floyd_${file}"
   vystupDiff="${rozdily}/diff_${metoda}_${file}"

   "$dijkstra" "${data}/${file}" > "${vystupD}"
   "$floyd" "${data}/${file}" > "${vystupF}"
   diff "${vystupD}" "${vystupF}" > "${vystupDiff}"
done

# Uklid souboru
make clean
[[ $? -ne 0 ]] && {
   echo "Chyba (make clean)! Nepodarilo se smazat vsechny vystupni soubory" >&2
   exit 2
}

exit 0

