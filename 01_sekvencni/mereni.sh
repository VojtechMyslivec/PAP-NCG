#!/bin/bash

# metoda vystupu z programu uzly | matice
metoda="uzly"

data="../data"
vystupy="./vystupy"
rozdily="$vystupy/diff"
dijkstra="./dijkstra"
floyd="./floyd-warshall"

# Pokud chybi Makefile, chyba a konec
[[ ! -r "Makefile" ]] && {
   echo "Chyba! Nelze provest prikaz make, chybi Makefile!" >&2
   exit 1
}

# Pokud neexistuji adresare, vytvori je
[[ ! -d "$vystupy" ]] && {
   mkdir "$vystupy"
}
[[ ! -d "$rozdily" ]] && {
   mkdir "$rozdily"
}


# Buildovani programu pres make
make

# Prava na spousteni spoustenych souboru
chmod +x "$dijkstra" "$floyd"

# Spousteni pro ruzne vstupni soubory
for file in `ls "$data"` ; do
   vystupD="${vystupy}/out_${metoda}_dijkstra_${file}"
   vystupF="${vystupy}/out_${metoda}_floyd_${file}"
   vystupDiff="${rozdily}/diff_${metoda}_${file}"

   "$dijkstra" "${data}/${file}" > "${vystupD}"
   "$floyd" "${data}/${file}" > "${vystupF}"
   diff "${vystupD}" "${vystupF}" > "${vystupDiff}"
done

# Uklid souboru
make clean


