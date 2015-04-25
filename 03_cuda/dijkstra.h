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
#include "funkceSpolecne.h"
#include "cDijkstra.h"

// alokuje data pro objekt cDijkstra na GPU. Inicializace hodnot dela jiz GPU
void dijkstraObjektInit( unsigned ** devGraf, unsigned pocetUzlu, unsigned idUzlu, cDijkstra *& devDijkstra );

// inicializuje a nakopiruje data grafu na GPU
void grafInicializaceNaGPU( unsigned ** graf, unsigned pocetUzlu, unsigned **& devGraf );

// vytvori pole ukazatelu na cDijkstra, ktere budou alokovane na GPU
void dijkstraInicializaceNaGPU( unsigned ** devGraf, unsigned pocetUzlu, cDijkstra **& devDijkstra );

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vola (paralelne) vypocet Dijkstrova algoritmu pro kazdy uzel
// idealni slozitost O( n^3 / p )
// vysledek vypise na stdout
bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken );

// funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& vzdalenostM, unsigned **& predchudceM, unsigned pocetVlaken );

// funkce, ktera zajisti uklizeni alokovane dvourozmerne promenne
void uklidUkazatelu( unsigned **& dveDimenze, unsigned rozmer );

// kopiruje data ze vsech objektu v poli devDijkstra na device do pole vzdalenostM na host
void zkopirujDataZGPU( unsigned ** vzdalenostM, cDijkstra ** devDijkstra, unsigned pocetUzlu );

// funkce pro vypis matice delek a predchudcu
void vypisVysledekMaticove( unsigned ** vzdalenosti, unsigned pocetUzlu );

