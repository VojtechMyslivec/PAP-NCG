/** floyd-warshall.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 algoritmus Floyd-Warshall
 *
 *
 */

#include "funkceSpolecne.h"
#include "cFloyd_Warshall.h"

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
/*
void floydWarshall( unsigned ** graf, unsigned pocetUzlu ) {   

   // inicializace ----------------------------------------
   unsigned ** delkaPredchozi  = new unsigned*[pocetUzlu];
   unsigned ** delkaAktualni   = new unsigned*[pocetUzlu];
   unsigned ** predchudcePredchozi = new unsigned*[pocetUzlu];
   unsigned ** predchudceAktualni  = new unsigned*[pocetUzlu];
   unsigned ** pomocny;
   unsigned novaVzdalenost;

   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      delkaPredchozi[i]  = new unsigned[pocetUzlu];
      delkaAktualni[i]   = new unsigned[pocetUzlu];
      predchudcePredchozi[i] = new unsigned[pocetUzlu];
      predchudceAktualni[i]  = new unsigned[pocetUzlu];
      for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
         delkaPredchozi[i][j]  = graf[i][j];
         if ( i == j || graf[i][j] == FW_NEKONECNO ) 
            predchudcePredchozi[i][j] = FW_NEDEFINOVANO;
         else
            predchudcePredchozi[i][j] = i;
      }
   }

   // vypocet ---------------------------------------------
   for ( unsigned k = 0 ; k < pocetUzlu ; k++ ) {
      for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
         for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
            // osetreni nekonecna
            if ( delkaPredchozi[i][k] == FW_NEKONECNO ||
                 delkaPredchozi[k][j] == FW_NEKONECNO )
               novaVzdalenost = FW_NEKONECNO;
            else
               novaVzdalenost = delkaPredchozi[i][k] + delkaPredchozi[k][j];

            // pokud nalezne kratsi cestu, zapise ji a zmeni predchudcePredchozi
            if ( novaVzdalenost < delkaPredchozi[i][j] ) {
               delkaAktualni[i][j]      = novaVzdalenost;
               predchudceAktualni[i][j] = predchudcePredchozi[k][j];
            }
            // jinak delka i predchudce zustavaji
            else {
               delkaAktualni[i][j]      = delkaPredchozi[i][j];
               predchudceAktualni[i][j] = predchudcePredchozi[i][j];
            }
         }
      }
      // prohozeni predchozi a aktualni
      pomocny        = delkaPredchozi;
      delkaPredchozi = delkaAktualni;
      delkaAktualni  = pomocny;
      pomocny             = predchudcePredchozi;
      predchudcePredchozi = predchudceAktualni;
      predchudceAktualni  = pomocny;
   }

   // vypis -----------------------------------------------
   vypisGrafu( cout,      delkaAktualni, pocetUzlu );
   cout << endl;
   vypisGrafu( cout, predchudceAktualni, pocetUzlu );

   // cleanup ---------------------------------------------
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      delete []      delkaPredchozi[i];
      delete []       delkaAktualni[i];
      delete [] predchudcePredchozi[i];
      delete []  predchudceAktualni[i];
   }
   delete []      delkaPredchozi;
   delete []       delkaAktualni;
   delete [] predchudcePredchozi;
   delete []  predchudceAktualni;

}*/

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

   cFloyd_Warshall* floyd_warshall = new cFloyd_Warshall( graf, pocetUzlu );
   floyd_warshall->spustVypocet();
   //floyd_warshall->vypisVysledekPoUzlech();
   floyd_warshall->vypisVysledekMaticove();
   
   delete floyd_warshall;
   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

