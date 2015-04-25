/** cDijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
 *                 trida cDijkstra pro Dijkstruv algoritmus
 *                    upraven vypocet prioritni fronty -- misto haldy for cylky, 
 *                    aby slo paralelizovat
 *
 *
 */

#include "cDijkstra.h"

#include <iostream>
#include <iomanip>

using namespace std;
// ============================================================================

cDijkstra::cDijkstra( unsigned idVychozihoUzlu ) {
    if ( idVychozihoUzlu >= pocetUzlu ) {
        cerr << "inicializace(): Chyba! id uzlu je vyssi nez pocet uzlu.";
        throw "inicializace(): Chyba! id uzlu je vyssi nez pocet uzlu.";
    }

    // vzdalenost = new unsigned[pocetUzlu];
    // uzavreny   = new bool[pocetUzlu];
    vzdalenost = NULL;
    uzavreny   = NULL;
    pocetUzavrenychUzlu   = 0;
    this->idVychozihoUzlu = idVychozihoUzlu;
}

cDijkstra::~cDijkstra() {
//     if (vzdalenost != NULL)
//         delete [] vzdalenost;
//     if (uzavreny != NULL)
//         delete [] uzavreny;
}

__device__ void cDijkstra::devInicializujHodnoty() {
    pocetUzavrenychUzlu = 0;
    for ( unsigned idUzlu = 0; idUzlu < pocetUzlu; idUzlu++ ) {
        vzdalenost[idUzlu] = DIJKSTRA_NEKONECNO;
        uzavreny[idUzlu] = false;
    }
    vzdalenost[idVychozihoUzlu] = 0;
}

__device__ bool cDijkstra::devSpustVypocet( ) {
    unsigned idUzlu;
    unsigned vzdalenostUzlu, vzdalenostSouseda, vzdalenostHrany, novaVzdalenost;

#ifdef DEBUG
    vypisFrontu( );
#endif // DEBUG

    // dokud nenavstivi vsechny uzly
    while ( pocetUzavrenychUzlu < pocetUzlu ) {
        // uzly navstevuje podle nejmensi vzdalenosti
        if ( ! vyberMinimumZFronty( idUzlu ) ) 
            return false;
        uzavreny[idUzlu] = true;
        pocetUzavrenychUzlu++;
#ifdef DEBUG
        cerr << "\nzpracovavam uzel " << idUzlu << endl;
        vypisFrontu( );
#endif // DEBUG
        vzdalenostUzlu = vzdalenost[idUzlu];
        // pokud je stale nekonecna, znamena to nedostupny uzel;
        if ( vzdalenostUzlu >= DIJKSTRA_NEKONECNO ) {
#ifdef DEBUG
            cerr << "preskakuji " << idUzlu << " vzd.: " << vzdalenostUzlu << endl;
#endif // DEBUG
            continue;
        }

        // pro vsechny sousedy, tedy for cyklus pres matici vzdalenosti
        for ( unsigned idSouseda = 0; idSouseda < pocetUzlu; idSouseda++ ) {
            // pokud je uzavreny, preskocim
            if ( uzavreny[idSouseda] == true ) {
                continue;
            }

            vzdalenostSouseda = vzdalenost[idSouseda];
            vzdalenostHrany = graf[idUzlu][idSouseda];
            // kontrola, aby nepretekla hodnota
            if ( vzdalenostHrany >= DIJKSTRA_NEKONECNO )
                novaVzdalenost = DIJKSTRA_NEKONECNO;
            else
                novaVzdalenost = vzdalenostUzlu + vzdalenostHrany;

            // nalezeni kratsi vzdalenosti
            if ( novaVzdalenost < vzdalenostSouseda ) {
#ifdef DEBUG
                cerr << "   nova vzdalenost z " << idUzlu << " do " << idSouseda << " = " << vzdalenostHrany << '(' << novaVzdalenost << ')' << endl;
#endif // DEBUG
                vzdalenost[idSouseda] = novaVzdalenost;

#ifdef DEBUG
                vypisFrontu( );
#endif // DEBUG
            }
        }
    }

    return true;
}

__device__ bool cDijkstra::devVyberMinimumZFronty( unsigned & idMinima ) const {
    if (pocetUzavrenychUzlu == pocetUzlu) {
        cerr << "vyberMinimumZFronty(): Chyba! Prazdna fronta!" << endl;
        return false;
    }

    unsigned minimum = DIJKSTRA_NEKONECNO;
    // pro kontrolu, zda se nejaky uzel najde
    idMinima = pocetUzlu;
    // ze vsech neuzavrenych uzlu vybere minimum
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
        if ( uzavreny[idUzlu] == true ) 
            continue;
        if ( vzdalenost[idUzlu] <= minimum ) {
            minimum  = vzdalenost[idUzlu];
            idMinima = idUzlu;
        }
    }
    // pro kontrolu, zda se nejaky uzel nasel
    if ( idMinima == pocetUzlu ) {
        cerr << "vyberMinimumZFronty(): Neocekavana chyba! Fronta neni prazdna, minimum se ale nenalezlo!" << endl;
        return false;
    }

    return true;
}

__device__ void cDijkstra::devVypisFrontu( ) const {
    printf( "Fronta:\n" );
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
        if      ( uzavreny[idUzlu] ) 
            printf( " z " );
        else if ( vzdalenost[idUzlu] == DIJKSTRA_NEKONECNO ) 
            printf( " - " );
        else
            printf( "%2d ", vzdalenost[idUzlu] );
    }
    printf( "\n" );
}

// unsigned * cDijkstra::getVzdalenost( ) const {
//     return this->vzdalenost;
// }
// 
// void cDijkstra::vypisVysledekPoUzlech( ) const {
//     unsigned hodnota;
//     cout << "id uzlu:         ";
//     for (unsigned i = 0; i < pocetUzlu; i++) {
//         cout << setw(2) << i << " ";
//     }
//     cout << "\n"
//         "Vzdalenosti[" << idVychozihoUzlu << "]:  ";
//     for (unsigned i = 0; i < pocetUzlu; i++) {
//         hodnota = vzdalenost[i];
//         if (hodnota == DIJKSTRA_NEKONECNO)
//             cout << " - ";
//         else
//             cout << setw(2) << hodnota << " ";
//     }
//     cout << endl;
// }

