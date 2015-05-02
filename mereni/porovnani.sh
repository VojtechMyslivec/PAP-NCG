#!/bin/bash
USAGE="USAGE
   $0 adresar_s_merenim_1 adresar_s_merenim_2 stitek

   Skript porovna dva aresare s vysledky s merenim (tedy
   adresar_s_merenim_1/vysledky a adresar_s_merenim_2/vysledky).
   Kde spoji dohromady 2 soubory se stejnym jmenem (dle poctu
   uzlu) ve formatu
   ---------------
   pocet_vlaken	prumerny_cas	zrychleni	efektivita
   ---------------

   Skript zkontroluje, ze pro oba soubory existuji zaznamy pro 
   stejne pocty vlaken a vystup zapise do souboru se stejnym 
   jmenem do (noveho) adresare urcenym stitkem ve formatu
   ---------------
   pocet_vlaken	pomer_casu	pomer_zrychleni	pomer_efektivity
   ---------------
   kde pomer_... je vzdy pomer dale veliciny z prvniho a z druheho 
   adresare.
"

varovani() {
   echo "$0: Varovani:" "$*" >&2
}
chyba() {
   echo "$0: Chyba:" "$*" >&2
}

[[ $1 == "-h" ]] && {
   echo "$USAGE"
   exit 0
}
[[ $# -eq 3 ]] || {
   echo "$USAGE"
   exit 1
}


adr1="$1/vysledky"
adr2="$2/vysledky"
# kontrola adresaru ------------------------------
[[ -d "$adr1" && -r "$adr1" ]] || {
   chyba "'$adr1' neni citelny adresar!"
   exit 1
}
[[ -d "$adr2" && -r "$adr2" ]] || {
   chyba "'$adr2' neni citelny adresar!"
   exit 1
}
# kontrola, zda existuje alespon jeden soubor
seznam=`ls "$adr1"`
[ `echo "$seznam" | wc -l` -ge 1 ] || {
   chyba "V adresari '$adr1' neni zadny soubor!"
   exit 2
}
# kontrola, zda existuji stejne se jmenujici soubory
echo "$seznam" | diff -q - <(ls "$adr2") 2> /dev/null || {
   chyba "V adresarich '$adr1' a '$adr2' nejsou soubory se stejnymi nazvy!"
   exit 2
}

# kontrola vystupniho adresare -------------------
cas=`date "+%Y-%m-%d_%H-%M-%S"`
adrPracovni="${3}"
[[ -d "$adrPracovni" ]] && {
   chyba "Adresar '$adrPracovni' jiz existuje!"
   exit 2
}
mkdir -p "$adrPracovni" 2> /dev/null || {
   chyba "Nelze vytvorit adresar '$adrPracovni'!"
   exit 2
}

old=$IFS
IFS="
"
for soubor in $seznam; do 
   # spojeni dvou serazenych souboru daneho jmena a vypocet pomeru v awk
   paste <(sort -n "$adr1/$soubor") <(sort -n "$adr2/$soubor") | awk '{
      if ( $1 != $5 )
         exit 1
      print $1 "\t" $2/$6 "\t" $3/$7 "\t" $4/$8
   }' > "$adrPracovni/$soubor" || { 
      chyba "Chyba pri zpracovani souboru '$soubor'. Pocet vlaken nesouhlasi."
      exit 3
   }
done

