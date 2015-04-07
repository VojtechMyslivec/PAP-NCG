#!/bin/bash
USAGE="USAGE
   $0 program soubor_s_grafem"

TIMEFORMAT=%R
merenyPocetVlaken="2 4 6 8 12 24"

# $1 -- program
# $2 -- pocet vlaken
# $3 -- soubor se vstupnimi daty
zmerVypocet() {
   "$1" -t "$2" -f "$3" 2>&1 >/dev/null | awk -F '\t' '{print $3}'
}

vypis() {
   echo "vlaken: $1	cas: $2	zrychleni: $3	efektivita: $4"
}

[[ -f "$1" && -x "$1" ]] || {
   echo "$USAGE"
   exit 1
}

[[ -f "$2" && -r "$2" ]] || {
   echo "$USAGE"
   exit 1
}

program=$1
data=$2

sekvencniCas=`zmerVypocet "$program" 1 "$data"`
vypis 1 "$sekvencniCas" 1 1

for pocet in $merenyPocetVlaken; do 
   cas=`zmerVypocet "$program" "$pocet" "$data"`
   [[ $cas =~ ^[0-9]+.[0-9]+$ ]]  || {
      echo "Vypocet skoncil s chybou (vycerpane prostredky)"
      exit 2
   }
   zrychleni=`echo "scale = 3; $sekvencniCas / $cas" | bc`
   efektivita=`echo "scale = 3; $zrychleni / $pocet" | bc`
   vypis "$pocet" "$cas" "$zrychleni" "$efektivita"
done

