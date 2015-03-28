/** floyd.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
 *                 algoritmus Floyd-Warshall, main
 *
 *
 */

#include "funkceSpolecne.h"
#include "floydWarshall.h"

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

// main =======================================================================
int main( int argc, char ** argv ) {
//   cout << "Hello Floyd-Warshall" << endl;
   unsigned ** graf          = NULL;
   char     *  souborSGrafem = NULL;
   unsigned    pocetUzlu     = 0;
   unsigned    pocetVlaken   = 5;
   unsigned    navrat;

   if ( parsujArgumenty( argc, argv, pocetVlaken, souborSGrafem, navrat ) != true ) {
      return navrat;
   }

   // nacteni dat
   if ( nactiData( souborSGrafem, graf, pocetUzlu ) != true ) {
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

   floydWarshall( graf, pocetUzlu, pocetVlaken );

   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

