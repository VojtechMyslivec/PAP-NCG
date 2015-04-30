/** dijkstra.cuh
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-duben 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 algoritmus Dijkstra
 *
 *
 */

#include "funkceSpolecne.cuh"
#include "cDijkstra.cuh"

// alokuje data pro objekt cDijkstra na GPU. Inicializace hodnot dela jiz GPU
void dijkstraObjektInit( unsigned ** devGraf, unsigned pocetUzlu, unsigned idUzlu, cDijkstra *& devDijkstra );

// vytvori pole ukazatelu na cDijkstra, ktere budou alokovane na GPU
void dijkstraInicializaceNaGPU( unsigned ** devGraf, unsigned pocetUzlu, cDijkstra **& devDijkstra );

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vola (paralelne) vypocet Dijkstrova algoritmu pro kazdy uzel
// idealni slozitost O( n^3 / p )
// vysledek vypise na stdout
bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu );

// TODO smazat // funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
// void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& vzdalenostM, unsigned **& predchudceM );

// funkce, ktera zajisti uklizeni alokovane dvourozmerne promenne
void uklidUkazatelu( unsigned **& dveDimenze, unsigned rozmer );

// kopiruje data ze vsech objektu v poli devDijkstra na device do pole vzdalenostM na host
void zkopirujDataZGPU( unsigned ** vzdalenostM, cDijkstra ** devDijkstra, unsigned pocetUzlu );

// kernel kod pro CUDA, wrapper obalujici spousteni vypoctu kazdeho z objektu v poli devDijkstra
__global__ void wrapperProGPU( cDijkstra ** devDijkstra, unsigned pocetUzlu, unsigned pocetWarpu );

