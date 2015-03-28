/** dijkstra.h
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
 *                 algoritmus Dijkstra
 *
 *
 */

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vola (paralelne) vypocet Dijkstrova algoritmu pro kazdy uzel
// idealni slozitost O( n^3 / p )
// vysledek vypise na stdout
bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken );

// funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& vzdalenostM, unsigned **& predchudceM, unsigned pocetVlaken );

// funkce, ktera zajisti uklizeni alokovane dvourozmerne promenne
void uklidUkazatelu( unsigned **& dveDimenze, unsigned rozmer );

// funkce pro vypis matice delek a predchudcu
void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka, unsigned ** predchudce );

