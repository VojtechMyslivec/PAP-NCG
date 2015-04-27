#!/bin/bash
USAGE="USAGE
   $0 program adresar_s_daty stitek [host]

      Pro vsechna data s koncovkou .txt v adresari_s_daty
      zaradi do fronty qsub program s temito parametry.

      Skript vytvori podadresar stitek s casovou znackou, do
      ktereho se budou ukladat soubory fronty a vystupy programu
      spoustenych pres frontu.

      Nepovinny parametr host urcuje pocitac, na kterem se
      uloha ve fronte bude spoustet
         povolene hodnoty jsou:
            gpu-02
            gpu-03  [vychozi]"


chyba () {
    echo "$0: Chyba:" "$*" >&2
}

typeset -A nahrazeni
nahrazeni["CHYBOVY"]=___NAHRAZENI-STDERR___
nahrazeni["STANDARDNI"]=___NAHRAZENI-STDOUT___
nahrazeni["PRIKAZ"]=___NAHRAZENI-PRIKAZ___

vzor="vzor.q"

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
[[ $# -eq 3 || $# -eq 4 ]] || {
    echo "$USAGE" >&2
    exit 1
}
[[ -f "$1" && -x "$1" ]] || {
    chyba "Soubor '$1' musi byt spustitelny program."
    exit 1
}
[[ -d "$2" && -r "$2" ]] || {
    chyba "Adresar '$2' musi byt citelny."
    exit 1
}


program=$1
data=$2
stitek=$3
host=${4:-gpu-03}
vstupy=`ls "$data/"*.txt 2> /dev/null`
# test, jestli existuje alespon jeden soubor
[[ -z "$vstupy" ]] && {
    chyba "V adresari '$data' neni zadny vstupni soubor s koncovkou .txt"
    exit 1
}
# test host
[[ "$host" == "gpu-02" || "$host" == "gpu-03" ]] || {
    chyba "Nepodporovany host!"
    exit 1
}

# Kontrola uzivatele o poctu uloh k vypocitani
souboru=`echo "$vstupy" | wc -l`

while true; do
    read -p "Zadat k vypoctu $souboru souboru (celkem $souboru uloh)? " odp
    if   [[ "$odp" =~ ^[aA]$  ]]; then
        break
    elif [[ "$odp" =~ ^[nN]$  ]]; then
        echo "Koncim"
        exit $OK
    else
        echo "Zadej a/n"
    fi
done

cas=`date "+%Y-%m-%d_%H-%M-%S"`
adrPracovni="${stitek}_${program##*/}_$cas"
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
   [[ "$n" =~ ^[0-9]+$ ]] || n="NaN"
   
   jmenoSouboru=${vstup##*/}

   fronta="fronta-n${n}-${jmenoSouboru%%.txt}.q"

   prikaz="\"${program}\" -f \"${vstup}\""
   sed  "s|${nahrazeni["PRIKAZ"]}|$prikaz|g
         s|${nahrazeni["CHYBOVY"]}|\"$adrMereni\"|g
         s|${nahrazeni["STANDARDNI"]}|\"$adrVystupy\"|g" "$vzor" > "$fronta"

   id=`qsub -l "h=${host}" "$fronta" 2> /dev/null | awk '{print $3}'`
   if [[ "$id" =~ ^[0-9]+$ ]]; then 
       echo "Pro soubor '$jmenoSouboru' zarazen vypocet s id ${id}."
   else
       echo "Pro soubor '$jmenoSouboru' se nepodarilo zaradit vypocet."
       id="NaN"
   fi
   mv "$fronta" "${adrFronty}/${fronta%.q}.${id}.q"

done

echo "Hotovo"

