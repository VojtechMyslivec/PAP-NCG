#!/bin/bash
USAGE="USAGE
   $0 soubor_s_grafem"
# $1 -- pocet vlaken
# $2 -- soubor se vstupnimi daty
zmerVypocet() {
   {
      time ./floyd-warshall -t "$1" -f "$2" >/dev/null
   } 2>&1
}

vypis() {
   echo "vlaken: $1	cas: $2	zrychleni: $3	efektivita: $4"
}

[[ -f "$1" && -r "$1" ]] || {
   echo "$USAGE"
   exit 1
}

data=$1
TIMEFORMAT=%R

sekvencniCas=`zmerVypocet 1 "$data"`
vypis 1 "$sekvencniCas" 1 1

for pocet in 2 3 4 5 8 10 15 20 25 30; do 
   cas=`zmerVypocet "$pocet" "$data"`
   [[ $cas =~ ^[0-9]+.[0-9]+$ ]]  || {
      echo "Vypocet skoncil s chybou (vycerpane prostredky)"
      exit 2
   }
   zrychleni=`echo "scale = 3; $sekvencniCas / $cas" | bc`
   efektivita=`echo "scale = 3; $zrychleni / $pocet" | bc`
   vypis "$pocet" "$cas" "$zrychleni" "$efektivita"
done

