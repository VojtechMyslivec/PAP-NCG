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
#include <cstring>

#define MAIN_OK            0
#define MAIN_ERR_VSTUP     1
#define MAIN_ERR_GRAF      2

// define pro floyd-warshall
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

// makro min ze dvou
#define min(a,b) ((a) < (b)) ? (a) : (b)

using namespace std;

// main =======================================================================
int main( int argc, char ** argv ) {
//   cout << "Hello Floyd-Warshall" << endl;
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

   floydWarshall( graf, pocetUzlu );

   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

