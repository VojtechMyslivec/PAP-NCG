/** main.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <zdenek.novy@fit.cvut.cz>,       FIT CVUT v Praze
 *              
 * Datum:       unor-kveten 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 Algoritmus Floyd-Warshall pomoci dlazdickovani
 *                 main
 *
 */

// Zvoli algoritmus pro preklad
// vychozi je dijkstra

#include "funkceSpolecne.hpp"
#include "floydWarshall.hpp"

#include <iostream>
//#include <fstream>
#include <iomanip>

using namespace std;

void mereni( unsigned ** graf, unsigned pocetUzlu ) {

    floydWarshall( graf, pocetUzlu );

}

// main =======================================================================
int main( int argc, char ** argv ) {
    unsigned ** graf          = NULL;
    char     *  souborSGrafem = NULL;
    unsigned    pocetUzlu     = 0;
    unsigned    navrat;

    if ( parsujArgumenty( argc, argv, souborSGrafem, navrat ) != true ) {
        return navrat;
    }

    // nacteni dat
    if ( nactiData( souborSGrafem, graf, pocetUzlu ) != true ) {
        uklid( graf, pocetUzlu );
        return MAIN_ERR_VSTUP;
    }

    // vypis a kontrola grafu
#ifdef VYPIS
    vypisGrafu( cout, graf, pocetUzlu );
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
    cout << endl;
#endif // VYPIS

    mereni( graf, pocetUzlu );

    uklid( graf, pocetUzlu );

    return MAIN_OK;
}

