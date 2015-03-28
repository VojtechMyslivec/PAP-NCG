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
//#include <string>
//#include <cstring>
#include <unistd.h> // getopts()
#include <cstdlib> // atoi()

#define MAIN_OK            0
#define MAIN_ERR_USAGE     1
#define MAIN_ERR_VSTUP     2
#define MAIN_ERR_GRAF      3
#define MAIN_ERR_NEOCEKAVANA 10

// define pro floyd-warshall
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

// makro min ze dvou
#define min(a,b) ((a) < (b)) ? (a) : (b)

using namespace std;

// souborSGrafem bude pointer na nejaky prvek pole argv, napr argv[2]
bool parsujArgumenty( int argc, char ** argv, unsigned & pocetVlaken, char *& souborSGrafem, unsigned & navrat ) {
   if ( argc < 2 ) {
      cerr << "Nedostatecny pocet argumentu" << endl;
      vypisUsage( cerr, argv[0] );
      navrat = MAIN_ERR_USAGE;
      return false;
   }

   int o = 0;
   souborSGrafem = NULL;
   int tmp;
   opterr = 0; // zabrani getopt, aby vypisovala chyby
   // + zajisti POSIX zadavani prepinacu, : pro rozliseni neznameho prepinace od chybejiciho argumentu
   while ( ( o = getopt( argc, argv, "+:hf:t:" ) ) != -1 ) {
      switch ( o ) {
         case 'h':
            vypisUsage( cout, argv[0] );
            navrat = MAIN_OK;
            return false;

         case 'f':
            souborSGrafem = optarg;
            break;

         case 't':
            tmp = atoi( optarg );
            if ( tmp < 1 ) {
               cerr << argv[0] << ": Pocet vlaken musi byt kladne cele cislo" << endl;
               navrat = MAIN_ERR_VSTUP;
               return false;
            }
            pocetVlaken = (unsigned)tmp;
            break;

         case ':':
            cerr << argv[0] << ": Prepinac '-" << (char)optopt << "' vyzaduje argument." << endl;
            navrat = MAIN_ERR_VSTUP;
            return false;

         case '?':
            cerr << argv[0] << ": Neznamy prepinac '-" << (char)optopt << "'." << endl;
            navrat = MAIN_ERR_VSTUP;
            return false;

         default:
            navrat = MAIN_ERR_NEOCEKAVANA;
            return false;
      }
   }
   if ( souborSGrafem == NULL ) {
      cerr << argv[0] << ": Musi byt urceno jmeno souboru s grafem (prepinac '-f')" << endl;
      navrat = MAIN_ERR_VSTUP;
      return false;
   }
   if ( optind != argc ) {
      cerr << argv[0] << ": Chyba pri zpracovani parametru, nejsou dovoleny zadne argumenty." << endl;
      navrat = MAIN_ERR_VSTUP;
      return false;
   }

   return true;
}

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

