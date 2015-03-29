#/bin/bash
# skript generuje grafy pro ucely semestralky PAP-NCG
# prepinaci lze urcit (ne)orientovane hrany, pocet uzlu
# a maximalni cenu hrany

maxCenaHrany=5
pocetUzlu=10
orientovany=false

USAGE="POUZITI

   $0 [ prepinace ]

      Skript generuje matice ohodnocenych hran grafu
      pro ucely semestralky PAP-NCG.
      Ve vychozim nastaveni generuje neorientovany graf 
      o ${pocetUzlu}-ti uzlech s maximalni cenou hrany ${maxCenaHrany}.
      
   prepinace:

      -h
          Vypise tuto napovedu a skonci.

      -n pocet_uzlu
          nastavi pocet uzlu generovaneho grafu 
          (Musi byt desitkove cislo)

      -c cena_hrany
          Nastavi maximalni cenu hrany.
          (Musi byt desitkove cislo)

      -o
          Generuje orientovany graf misto neorientovaneho.

PRIKLADY
   vychozi nastaveni
      $0

   orientovany graf o 20-ti uzlech s max. cenou hrany 100
      $0 -on 20 -c 100 

   pro matici hran
      $0 -c 1"

chyba() {
   echo "$*" >&2
}

# argument $1 porovna jako desitkove cislo, pokud selze
# vypise chybu s nazvem parametru jako $2 a skonci 
# exit $3
cislo() {
   [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
      chyba "$2 musi byt cele kladne desitkove cislo!"
      exit $3
   }
}

# zpracovani prikazove radky
# + pro POSIX, : pro potlaceni chybovych hlasek
while getopts ":n:c:oh" OPT; do
   case $OPT in
      n) 
         cislo "$OPTARG" "Pocet uzlu" 2
         pocetUzlu=$OPTARG
         ;;
      c)
         cislo "$OPTARG" "Cena hrany" 2
         maxCenaHrany=$OPTARG
         ;;
      o)
         orientovany=true
         ;;
      h)
         echo "$USAGE" 
         exit 0;
         ;;
      *)
         chyba "$0: Neplatny prepinac '-$OPTARG'"
         exit 1;
         ;;
   esac
done
shift $(( OPTIND - 1 ))

[[ $# -eq 0 ]] || {
   chyba "$USAGE"
   exit 1
}

cifer=${#maxCenaHrany}
typeset -A matice

echo "$pocetUzlu"
for (( i = 1 ; i <= pocetUzlu ; i++ )); do 
   echo -n "   "
   for (( j = 1 ; j <= pocetUzlu ; j++ )); do 
      # pokud je to hrana z i do i, je nulova
      if   [[ $i -eq $j ]]; then
         cenaHrany=0
      # neorientovana hrana, ktera jiz byla vygenerovana
      elif [[ "$orientovany" == false && $j -lt $i ]]; then
         cenaHrany=${matice[$j,$i]}
      # nova hrana -- 50%, ze bude pritomna
      elif (( RANDOM % 2 == 0 )); then
         cenaHrany=$(( (RANDOM % maxCenaHrany) + 1 ))
      # nova hrana -- neexistuje
      else
         cenaHrany=-;
      fi

      # pokud je neorientovany, je potreba si pamatovat hrany
      [[ "$orientovany" == false ]] && {
         matice[$i,$j]=$cenaHrany
      }

      printf "%${cifer}s " "$cenaHrany"
   done
   echo
done

