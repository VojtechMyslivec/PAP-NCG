#!/bin/bash
USAGE="USAGE
   $0 jmeno_adresare
   
   jmeno_adresare   adresar, ktery bude vytvoren s novou sadou 
                    vygenerovanych grafu"

generator=./grafGenerator.sh
sekvenceN=5000 #`seq 4000 1000 10000`
pocetGrafu=5

maximalniCenaHrany=9
pravdepodobnostHrany=50
#prepinacOrientace=""
prepinacOrientace="-o"

chyba() {
   echo "$*" >&2
}

[[ -f "$generator" && -x "$generator" ]] || {
   chyba "Chybi skript pro generovani grafu '$generator'"
   exit 2
}

[[ $# -eq 1 ]] || {
   chyba "$USAGE"
   exit 1
}

adr=$1
[[ -d "$adr" ]] || {
   echo "Vytvarim adresar '$adr'"
   mkdir -p "$adr" 2>/dev/null || {
      chyba "Nelze vytvorit adresar '$adr'"
      exit 2
   }
}
[[ -w "$adr" ]] || {
   chyba "Nelze zapisovat do '$adr'!"
   exit 2
}

for n in $sekvenceN; do
   echo "Generuji graf pro n = $n"
   echo -n "   "
   for i in `seq "$pocetGrafu"`; do 
      vystupniSoubor="${adr}/graf-n${n}${prepinacOrientace}-${i}.txt"
      # prepinac je schvalne bez uvozovek!
      "$generator" $prepinacOrientace -c "$maximalniCenaHrany" -n "$n" > "$vystupniSoubor" &
      echo -n "."
   done
   echo
done

