#!/bin/bash
USAGE="USAGE
   $0 program adresar_s_daty

      Pro vsechna data s koncovkou .txt v tomto adresari_s_daty
      a pro vsechny merene pocty vlaken zaradi do fronty qsub 
      program s temito parametry
"
chyba () {
   echo "$0: Chyba:" "$*" >&2
}

typeset -A nahrazeni
nahrazeni["CHYBOVY"]=___NAHRAZENI-STDERR___
nahrazeni["STANDARDNI"]=___NAHRAZENI-STDOUT___
nahrazeni["PRIKAZ"]=___NAHRAZENI-PRIKAZ___

vzor="vzor.q"
merenyPocetVlaken="1 2 4 6 8 12 24"

[[ -f "$vzor" && -r "$vzor" ]] || {
   chyba "Soubor se vzorovou frontou '$vzor' neni citelny"
   exit 2
}
for hodnota in "${nahrazeni[@]}"; do
   grep -q "$hodnota" "$vzor" || {
      chyba "Vzorovy soubor nema zastupny symbol '$hodnota'"
      exit 2
   }
done

[[ "$1" == "-h" ]] && {
   echo "$USAGE"
   exit 0
}
[[ $# -eq 2 ]] || {
   echo "$USAGE" >&2
   exit 1
}
[[ -f "$1" && -x "$1" ]] || {
   chyba "Soubor '$1' musi byt spustitelny program"
   exit 1
}
[[ -d "$2" && -r "$2" ]] || {
   chyba "Adresar '$2' musi byt citelny"
   exit 1
}

program=$1
data=$2
vstupy=`ls "$data/"*.txt 2> /dev/null`
# test, jesltli existuje alespon jeden soubor
[[ -z "$vstupy" ]] && {
   chyba "V adresari '$data' neni zadny vstupni soubor s koncovkou .txt"
   exit 1
}

# Kontrola uzivatele o poctu uloh k vypocitani
souboru=`echo "$vstupy" | wc -l`
ulohNaSoubor=`echo "$merenyPocetVlaken" | wc -w`

while true; do
   read -p "Zadat k vypoctu $souboru souboru po $ulohNaSoubor ulohach, celkem $((souboru*ulohNaSoubor)) uloh? " odp
   if   [[ "$odp" =~ ^[aA]$  ]]; then
      break;
   elif [[ "$odp" =~ ^[nN]$  ]]; then
      echo "Koncim"
      exit $OK
   else
      echo "Zadej a/n"
   fi
done

cas=`date "+%Y-%m-%d_%H-%M-%S"`
adrPracovni="${program##*/}_$cas"
adrVystupy="$adrPracovni/vystupy"
adrMereni="$adrPracovni/mereni"
adrFronty="$adrPracovni/fronty"

mkdir -p "$adrVystupy" "$adrMereni" "$adrFronty" || {
   chyba "nelze vytvorit pracovni adresare '$adrPracovni'"
   exit 2
}

# Cyklus pres vsechny vstupni soubory v adresari data
oldIFS=$IFS;
IFS="
"
for vstup in $vstupy ; do
   n=${vstup##*n}
   n=${n%%[-.]*}
   [[ "$n" =~ ^[0-9]+$ ]] || n=NaN
   
   jmenoSouboru=${vstup##*/}

   echo -n "Pro soubor '$vstup' zarazen vypocet ( vlaken "
   IFS=$oldIFS
   for t in $merenyPocetVlaken; do
      [[ "$t" =~ ^[0-9]+$ ]] || {
         chyba "Pocet vlaken '$t' neni cislo" 
         exit 2
      }

      fronta="fronta-n${n}-t${t}-${jmenoSouboru%%.txt}.q"
      prikaz="\"${program}\" -t \"${t}\" -f \"${vstup}\""
      sed  "s|${nahrazeni["PRIKAZ"]}|$prikaz|g
            s|${nahrazeni["CHYBOVY"]}|\"$adrMereni\"|g
            s|${nahrazeni["STANDARDNI"]}|\"$adrVystupy\"|g" "$vzor" > "$fronta"

      id=`qsub -l h=gpu-02 "$fronta" 2> /dev/null | awk '{print $3}'`
      if [[ "$id" =~ ^[0-9]+$ ]]; then 
         echo -n "$t "
      else
         echo -n "XX"
         id="NaN"
      fi
      mv "$fronta" "${adrFronty}/${fronta%.q}.${id}.q"

   done

   echo ")"

done
echo Done

