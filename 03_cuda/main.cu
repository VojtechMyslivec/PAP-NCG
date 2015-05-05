/** main.cu
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-kveten 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 main
 *
 */

// Zvoli algoritmus pro preklad
// vychozi je dijkstra

#ifndef DIJKSTRA
   #define DIJKSTRA
#endif // DIJKSTRA

#ifdef FLOYDWARSHALL
   #undef DIJKSTRA
#endif // FLOYDWARSHALL
// zbytecne, ale symetricke
// #ifdef DIJKSTRA
//    #undef FLOYDWARSHALL
// #endif // DIJKSTRA

#include <iostream>

#include "funkceSpolecne.cuh"
#ifdef FLOYDWARSHALL
   #include "floydWarshall.cuh"
#endif // FLOYDWARSHALL
#ifdef DIJKSTRA
   #include "dijkstra.cuh"
#endif // DIJKSTRA

using namespace std;

// mereni jiz v danem algoritmu pomoci CUDA udalosti
bool mereni( unsigned ** graf, unsigned pocetUzlu, unsigned velikostMatice, unsigned pocetWarpu ) {

#ifdef DIJKSTRA
    return dijkstraNtoN(  graf, pocetUzlu, pocetWarpu );
#endif // DIJKSTRA

#ifdef FLOYDWARSHALL
    return floydWarshall( graf, pocetUzlu, velikostMatice, pocetWarpu );
#endif // FLOYDWARSHALL

}

// main =======================================================================
int main( int argc, char ** argv ) {
    //TODO 
    HANDLE_ERROR( cudaSetDevice( 1 ) );
    unsigned ** graf           = NULL;
    char     *  souborSGrafem  = NULL;
    unsigned    pocetWarpu     = CUDA_VYCHOZI_POCET_WARPU;
    unsigned    pocetUzlu      = 0;
    unsigned    velikostMatice = 0;
    unsigned    navrat;

    if ( parsujArgumenty( argc, argv, souborSGrafem, pocetWarpu, navrat ) != true ) {
        return navrat;
    }

    // nacteni dat
    if ( nactiData( souborSGrafem, graf, pocetUzlu, velikostMatice ) != true ) {
        uklid( graf, pocetUzlu );
        return MAIN_ERR_VSTUP;
    }

    // vypis a kontrola grafu
#ifdef VYPIS
    vypisGrafu( cout, graf, pocetUzlu );
#endif // VYPIS
    switch ( kontrolaGrafu( graf, pocetUzlu ) ) {
        case GRAF_NEORIENTOVANY:
#ifdef VYPIS
            cout << "Graf je neorientovany" << endl;
#endif // VYPIS
            break;
        case GRAF_ORIENTOVANY:
#ifdef VYPIS
            cout << "Graf je orientovany" << endl;
#endif // VYPIS
            break;
        case GRAF_CHYBA:
        default:
            return MAIN_ERR_GRAF;
    }
#ifdef VYPIS
    cout << endl;
#endif // VYPIS

    bool bNavrat;
    bNavrat = mereni( graf, pocetUzlu, velikostMatice, pocetWarpu );

    uklid( graf, pocetUzlu );

    if ( bNavrat != true ) 
        return MAIN_ERR_VYPOCET;
    return MAIN_OK;
}

