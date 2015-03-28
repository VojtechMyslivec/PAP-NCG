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
#include <unistd.h>
#include <stdlib.h>

#define MAIN_OK            0
#define MAIN_ERR_VSTUP     1
#define MAIN_ERR_GRAF      2

// define pro floyd-warshall
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

// makro min ze dvou
#define min(a,b) ((a) < (b)) ? (a) : (b)

using namespace std;

// Mozne parametry -h -f soubor -t pocet_vlaken
bool parsujArgumenty( int argc, char ** argv, unsigned & pocetVlaken, char *& souborSGrafem ) {
   int o = 0;

   if ( argc < 2 ) {
      cerr << "Nedostatecny pocet parametru" << endl;
      vypisUsage(cerr, argv[0]);
      return false;
   }

   while ( (o = getopt( argc, argv, "hf:t:" )) != -1 )
   switch (o) {
      case 'h':
         vypisUsage( cout, argv[0] );
         exit( MAIN_OK );
      case 'f':
         souborSGrafem = optarg;
         break;
      case 't':
         if ( atoi(optarg) < 0 ) {
            cerr << "Pocet vlaken nemuze byt mensi nez 0" << endl;
            return false;
         }
         pocetVlaken = atoi(optarg);
         break;
      case '?':
         if (optopt == 'c')
            cerr << "Option -" << optopt << " requires an argument." << endl;
         else if ( isprint (optopt) )
            cerr << "Unknown option '-" << optopt << "'." << endl;
         else
            cerr << "Unknown option character '\\x" << optopt << "'." << endl;
         vypisUsage( cerr, argv[0] );
         return false;
      default:
         abort( );
   }
   return true;
}

// main =======================================================================
int main( int argc, char ** argv ) {
//   cout << "Hello Floyd-Warshall" << endl;
   unsigned ** graf      = NULL;
   char     *  souborSGrafem = NULL;
   unsigned    pocetUzlu = 0;
   unsigned    pocetVlaken = 5;

   parsujArgumenty( argc, argv, pocetVlaken, souborSGrafem );
   /*
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
   }*/

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

   floydWarshall( graf, pocetUzlu, pocetVlaken );

   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

