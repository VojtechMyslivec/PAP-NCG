#!/bin/bash
USAGE="USAGE
    $0 adresar_s_merenim adresar_se_sekvencnimi_vysledky
 
    Skript vytvori vysledky ze souboru *.e* v podadresari 
    adresar_s_merenim/mereni, kde obsah souboru ma format
    ---------------
    pocet_uzlu	pocet_bloku	pocet_vlaken_v_bloku	cas_vypoctu	celkovy_cas
    ---------------
    Z adresar_se_sekvencnimi_vysledky (viz openMP-vysledky.sh) vyhleda
    prumerny cas sekvencniho reseni pro dany pocet uzlu (je pouzito
    pro vypocet zrychleni).

    Vysledky ulozi do podadresare adresar_s_merenim/vysledky
    do souboru vysledky-pocet_uzlu.txt ve formatu
    ---------------
    pocet_bloku	pocet_vlaken_v_bloku	prumerny_cas_vypoctu	prumerny_celkovy_cas	zrychleni
    ---------------
    Kde zrychleni  = (prumerny cas sekvencniho reseni) / (prumerny_celkovy_cas)
"

predpona="vysledky-"
pripona=".txt"

varovani() {
    echo "$0: Varovani:" "$*" >&2
}
chyba() {
    echo "$0: Chyba:" "$*" >&2
}
citelnyAdresar() {
    [[ -d "$1" && -r "$1" ]] 
}

[[ $1 == "-h" ]] && {
    echo "$USAGE"
    exit 0
}
[[ $# -eq 2 ]] || {
    echo "$USAGE"
    exit 1
}
adr=$1
adrSekvencni=$2
adrMereni=$adr/mereni
adrVysledky=$adr/vysledky

citelnyAdresar "$adr" || {
    chyba "'$adr' neni citelny adresar!"
    exit 1
}

citelnyAdresar "$adrMereni" || {
    chyba "Podadresar '$adrMereni' neexistuje ci neni citelny!" 
    exit 2
}

if [[ -e "$adrVysledky" ]]; then
    if citelnyAdresar "$adrVysledky"; then
        varovani "Podadresar '$adrVysledky' uz existuje."
    else
        chyba "Soubor '$adrVysledky' musi byt citelny adresar!"
        exit 2
    fi
else
    mkdir -p "$adrVysledky" || {
        chyba "Nelze vytvorit podadresar '$adrVysledky'!"
        exit 2
    }
fi

citelnyAdresar "$adrSekvencni" || {
    chyba "'$adrSekvencni' neni citelny adresar"
    exit 1
}
ls "$adrSekvencni/$predpona"*"$pripona" >/dev/null 2>&1 || {
    chyba "V '$adrSekvencni' nexeistuji data s vysledky!"
    exit 1 
}

cat "$adrMereni"/*.e* | awk \
   -v adr="$adrVysledky" -v adrSekvencni="$adrSekvencni" -v predpona="$predpona" -v pripona="$pripona" '
# funkce nalezne sekvencni cas pro dane n ze souboru v adrSekvencni
#   funkce nastavi nalezene[n] na 1 a sekvencniCas[n] na cas
#   ze souboru nebo na 0, pokud soubor nebo sekv. reseni neexistuje 
function najdiSekvencniCas( n ) {
    # aby se znovu pro stejne n nehledalo
    nalezene[n] = 1

    # soubor pro dane n
    vstupniSoubor = adrSekvencni "/" predpona n pripona
    while ( ( getline radek < vstupniSoubor ) > 0 ) {
        split( radek, sloupce )
        # pokud je v prvnim sloupci 1, jedna se o sekvencni reseni
        if ( sloupce[1] == 1 ) {
            sekvencniCas[n] = sloupce[2]
            return
        }
    }

    # pokud cyklus skonci, znamena to, ze soubor nebo sekv. reseni neexistuje
    sekvencniCas[n] = 0
}

# funkce, ktera vraci sekvencni cas
#   vrati 0, pokud soubor nebo sekv. reseni pro dane n neexistuje
function sekvencni( n ) {
    # pokud uz sekvencni cas pro dane n nalezl, nehleda znovu
    if ( nalezene[n] == 0 )
        najdiSekvencniCas( n )
    
    return sekvencniCas[n]
}

{
    i = $1 " " $2 " " $3
    cas[i]        += $4
    celkovyCas[i] += $5
    pocet[i]++
}

END {
    # pro celkovou statistiku dle poctu uzlu
    for ( j in cas ) {
        split( j, hodnota, " " )

        uzlu        = hodnota[1]
        bloku       = hodnota[2]
        vlaken      = hodnota[3]
        prumernyCas = cas[j] / pocet[j]
        prumernyCelkovyCas = celkovyCas[j] / pocet[j]

        zrychleni   = sekvencni( uzlu ) / prumernyCelkovyCas

        soubor = adr "/" predpona uzlu pripona
        print bloku "\t" vlaken "\t" prumernyCas "\t" prumernyCelkovyCas "\t" zrychleni   >> soubor
        close( soubor )
    }
}'



