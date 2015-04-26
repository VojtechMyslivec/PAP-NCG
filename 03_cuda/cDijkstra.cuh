/** cDijkstra.cuh
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-duben 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 trida cDijkstra pro Dijkstruv algoritmus
 *                    upraven vypocet prioritni fronty -- misto haldy for cylky, 
 *                    aby slo paralelizovat
 *
 *
 */

#ifndef CDIJKSTRA_kljnef29kjdsnf02
#define CDIJKSTRA_kljnef29kjdsnf02

#ifdef DEBUG2
    #define DEBUG
#endif // DEBUG2

#include "funkceSpolecne.cuh"
#include <climits>

#define DIJKSTRA_NEKONECNO    UNSIGNED_NEKONECNO

// ============================================================================
// trida pro vypocet nejkratsich cest od jednoho uzlu ke vsem ostatnim
// jako vnitri pomocnou strukturu pouziva binarni haldu implementovanou polem

class cDijkstra {
    public:
        // POZOR zadna alokace ani nastaveni vnitrnich poli nejsou inicializovana
        cDijkstra( unsigned pocetUzlu, unsigned idVychozihoUzlu );

        // priprava na vytvoreni pseudo-staickeho pointeru na graf
        // alokuje a nakopiruje graf do pameti GPU
        static void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& devGraf );
        ~cDijkstra( );

        // nastavi hodnoty poli a promennych dle jiz nastavenych hodnot 
        // pocetUzlu a idVychozihoUzlu z konstruktoru
        __device__ void devInicializujHodnoty( );

        // Vypocte nejkratsi cesty od daneho uzlu ke vsem ostatnim uzlum
        // Fronta hran je implementovana prostym prohledavanim pole (uzly
        // jsou oznacene jako otevrene/zavrene, zda se ve fronte jeste 
        // vyskytuji ci nikoliv
        //
        // Slozitost vypoctu 1 -> n je O( |H|.|U| + |U|.|U| ) = 
        // = O( |U|^3 ).
        __device__ bool devSpustVypocet( );

//         // navrat poli s vysledky
//         unsigned * getPredchudce( ) const;
//         unsigned * getVzdalenost( ) const;
// 
//         void vypisVysledekPoUzlech( ) const;

//    protected:
        // nalezne mimimum z fronty (neboli z otevrenych uzlu)
        // false v pripade prazdne fronty
        __device__ bool devVyberMinimumZFronty( unsigned & idMinima ) const;
        __device__ void devVypisFrontu( ) const;

        // ukazatel na matici vzdalenosti -- ohodnocenych hran
        // POZOR !! melka kopie !!
        // static unsigned ** graf;
        // static unsigned    pocetUzlu;
        // TODO pseudo-static
        unsigned ** graf;

        unsigned pocetUzlu;
        unsigned idVychozihoUzlu;

        unsigned * vzdalenost;
        // jestli je jiz cesta k uzlu vytvorena
        // zaroven urcije, jestli je jeste v prioritni fronte
        bool     * uzavreny;
        unsigned   pocetUzavrenychUzlu;
};

#endif // CDIJKSTRA_kljnef29kjdsnf02

