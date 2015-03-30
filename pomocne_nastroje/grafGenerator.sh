#/bin/bash
# skript generuje grafy pro ucely semestralky PAP-NCG
# prepinaci lze urcit (ne)orientovane hrany, pocet uzlu
# a maximalni cenu hrany

maxCenaHrany=5
pocetUzlu=10
orientovany=false
pravdepodobnost=50

USAGE="POUZITI

   $0 [ prepinace ]

      Skript generuje matice ohodnocenych hran grafu
      pro ucely semestralky PAP-NCG.
      
   prepinace:

      -h
          Vypise tuto napovedu a skonci.

      -n pocet_uzlu
          Nastavi pocet uzlu generovaneho grafu 
          Vychozi hodnota: $pocetUzlu
          (Musi byt desitkove cislo)

      -c cena_hrany
          Nastavi maximalni cenu hrany.
          Vychozi hodnota: $maxCenaHrany
          (Musi byt desitkove cislo)

      -p pravdepodobnost
          Nastavi pravdepodobnost existence hrany v procentech
          Vychozi hodnota: $pravdepodobnost
          (Musi byt desitkove cislo)

      -o
          Generuje orientovany graf misto neorientovaneho.

PRIKLADY
   vychozi nastaveni
      $0

   orientovany graf o 20-ti uzlech s max. cenou hrany 100
      $0 -on 20 -c 100 

   ridky graf o 100 uzlech
      $0 -p 3 -n 100

   matice hran
      $0 -c 1"

varovani() {
   echo "$0: Varovani:" "$*" >&2
}

chyba() {
   echo "$0: Chyba:" "$*" >&2
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
while getopts ":n:c:p:oh" OPT; do
   case $OPT in
      n) 
         cislo "$OPTARG" "Pocet uzlu" 2
         pocetUzlu=$OPTARG
         ;;
      c)
         cislo "$OPTARG" "Cena hrany" 2
         maxCenaHrany=$OPTARG
         ;;
      p)
         cislo "$OPTARG" "Pravdepodobnost" 2
         pravdepodobnost=$OPTARG
         [[ "$pravdepodobnost" -gt 100 ]] && {
            varovani "Pravdepodobnost je vetsi nez 100 %."
         }
         ;;
      o)
         orientovany=true
         ;;
      h)
         echo "$USAGE" 
         exit 0;
         ;;
      *)
         chyba "Neplatny prepinac '-$OPTARG'"
         exit 1;
         ;;
   esac
done
shift $(( OPTIND - 1 ))

[[ $# -eq 0 ]] || {
   echo "$USAGE" >&2
   exit 1
}

cifer=${#maxCenaHrany}
typeset -A matice

echo "$pocetUzlu"
#for (( i = 0 ; i < pocetUzlu ; i++ )); do
for i in `seq "$pocetUzlu"`; do 
   echo -n "   "
   #for (( j = 0 ; j < pocetUzlu ; j++ )); do
   for j in `seq "$pocetUzlu"`; do 
      # pokud je to hrana z i do i, je nulova
      if   [[ $i -eq $j ]]; then
         cenaHrany=0
      # neorientovana hrana, ktera jiz byla vygenerovana
      elif [[ "$orientovany" == false && $j -lt $i ]]; then
         cenaHrany=${matice[$j,$i]}
      # nova hrana -- 50%, ze bude pritomna
      elif (( RANDOM % 100 < pravdepodobnost )); then
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

