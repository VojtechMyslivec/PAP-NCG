/** dijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 algoritmus Dijkstra
 *
 *
 */

#include "cDijkstra.h"
#include "funkceSpolecne.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

#define MAIN_OK            0
#define MAIN_ERR_VSTUP     1
#define MAIN_ERR_GRAF      2

using namespace std;

void zkopirujPole(unsigned *from, unsigned **to, unsigned index, unsigned pocetUzlu) {
    for (unsigned i=0 ; i<pocetUzlu ; i++) {
        to[index][i] = from[i];
    }
}

void dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu ) {
   cDijkstra * dijkstra = new cDijkstra( graf, pocetUzlu );
   
   unsigned *pomocny;
   unsigned ** mPredchudcu = new unsigned*[pocetUzlu];
   unsigned ** mVzdalenosti = new unsigned*[pocetUzlu];
   for (unsigned i=0 ; i<pocetUzlu ; i++) {
       mPredchudcu[i] = new unsigned[pocetUzlu];
       mVzdalenosti[i] = new unsigned[pocetUzlu];
   }

   
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      //cout << "\nDijkstra pro uzel id = " << i << endl;      
      pomocny = dijkstra->spustVypocet( i );
      if (pomocny==NULL) {
          cerr << "problem s vypoctem pro id = " << i << endl;
      } else {
        zkopirujPole(pomocny, mVzdalenosti, i, pocetUzlu);
      }
   }
   
   // dijkstra->vypisVysledekPoUzlech( i );
   //dijkstra->vypisVysledekMaticove();
   cout << "Vzdalenosti:" << endl;
   vypisGrafu(cout, mVzdalenosti, pocetUzlu);
   
   delete dijkstra;
   
   for (unsigned i=0 ; i<pocetUzlu ; i++) {
       delete [] mPredchudcu[i];
       delete [] mVzdalenosti[i];
   }
   delete [] mPredchudcu;
   delete [] mVzdalenosti;
}

// main =======================================================================
int main( int argc, char ** argv ) {
//   cout << "Hello Dijkstra" << endl;
   unsigned ** graf      = NULL;
   unsigned    pocetUzlu = 0;

   if ( argc != 2 ) {
      vypisUsage( cerr, argv[0] );
      return MAIN_ERR_VSTUP;
   }
   // help ?
   if ( 
         strncmp( argv[1], PARAMETER_HELP1, sizeof(PARAMETER_HELP1) ) == 0 || 
         strncmp( argv[1], PARAMETER_HELP2, sizeof(PARAMETER_HELP2) ) == 0 
      ) {
      vypisUsage( cout, argv[0] );
      return MAIN_OK;
   }

   // nacteni dat
   if ( nactiData( argv[1], graf, pocetUzlu ) != true ) {
      uklid( graf, pocetUzlu );
      return MAIN_ERR_VSTUP;
   }

   // vypis a kontrola grafu
   vypisGrafu( cout, graf, pocetUzlu );
   switch ( kontrolaGrafu( graf, pocetUzlu ) ) {
      case GRAF_NEORIENTOVANY:
         cout << "Graf je neorientovany" << endl;
         break;
      case GRAF_ORIENTOVANY:
         cout << "Graf je orientovany" << endl;
         break;
      case GRAF_CHYBA:
      default:
         return MAIN_ERR_GRAF;
   }
   cout << endl;
   
   // n krat volany Dijkstra
   dijkstraNtoN( graf, pocetUzlu );

   uklid( graf, pocetUzlu );
   return MAIN_OK;
}

