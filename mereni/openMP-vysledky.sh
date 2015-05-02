#!/bin/bash
USAGE="USAGE
   $0 adresar_s_merenim

   Skript vytvori vysledky ze souboru *.e* v podadresari 
   adresar_s_merenim/mereni, kde obsah souboru ma format 
   ---------------
   pocet_uzlu	pocet_vlaken	cas
   ---------------
   Vysledky ulozi do podadresare adresar_s_merenim/vysledky
   do souboru vysledky-pocet_uzlu.txt ve formatu
   ---------------
   pocet_vlaken	prumerny_cas	zrychleni	efektivita
   ---------------
   Kde zrychleni  = (prumerny_cas pro 1 vlakno) / (prumerny_cas)
       efektivita =                 (zrychleni) / (pocet_vlaken)

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
[[ $# -eq 1 ]] || {
    echo "$USAGE"
    exit 1
}
adr=$1
adrMereni=$adr/mereni
adrVysledky=$adr/vysledky
[[ -d "$adr" && -r "$adr" ]] || {
    chyba "'$adr' neni citelny adresar!"
    exit 1
}
[[ -d "$adrMereni" && -r "$adrMereni" ]] || {
    chyba "Podadresar '$adrMereni' neexistuje ci neni citelny!" 
    exit 1
}
[[ -e "$adrVysledky" && ! ( -d "$adrVysledky" && -r "$adrVysledky" ) ]] && {
    chyba "Soubor '$adrVysledky' musi byt citelny adresar!"
}
if [[ -d "$adrVysledky" ]]; then
    varovani "Podadresar '$adrVysledky' uz existuje."
else
    mkdir -p "$adrVysledky" || {
       chyba "Nelze vytvorit podadresar '$adrVysledky'!"
       exit 1
    }
fi

cat "$adrMereni"/*.e* | awk \
   -v adr="$adrVysledky" -v predpona="vysledky-" -v pripona=".txt" '
{
    i = $1 " " $2
    cas[i] += $3
    pocet[i]++
}

END {
    # pro predvypocet prumeru sekvencniho casu
    for ( j in cas ) {
        split( j, hodnota, " " )
        if ( hodnota[2] != 1 )
            continue
 
        uzlu            = hodnota[1]
        sekvencni[uzlu] = cas[j] / pocet[j]
    }
 
    # pro celkovou statistiku dle poctu uzlu
    for ( j in cas ) {
        split( j, hodnota, " " )
 
        uzlu       = hodnota[1]
        vlaken     = hodnota[2]
        prumer     = cas[j] / pocet[j]
 
        zrychleni  = sekvencni[uzlu] / prumer
        efektivita = zrychleni / vlaken
 
        soubor = adr "/" predpona uzlu pripona
        print vlaken "\t" prumer "\t" zrychleni "\t" efektivita   >> soubor
        close( soubor )
    }
}'



