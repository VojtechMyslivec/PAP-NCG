/** dijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
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
#include <omp.h>

#define MAIN_OK            0
#define MAIN_ERR_VSTUP     1
#define MAIN_ERR_GRAF      2

using namespace std;

void inicializaceNtoN( unsigned ** graf, unsigned pocetUzlu, 
                       unsigned **& vzdalenostM, unsigned **& predchudceM ) {
   // inicializace matic vysledku
   vzdalenostM = new unsigned*[pocetUzlu];
   predchudceM = new unsigned*[pocetUzlu];
   for ( unsigned i = 0; i < pocetUzlu; i++ ) {
      vzdalenostM[i] = new unsigned[pocetUzlu];
      predchudceM[i] = new unsigned[pocetUzlu];
      for ( unsigned j = 0; j < pocetUzlu; j++ ) {
         vzdalenostM[i][j] = DIJKSTRA_NEKONECNO;
         predchudceM[i][j] = DIJKSTRA_NEDEFINOVANO;
      }
   }

   // staticka inicializace
   cDijkstra::inicializace( graf, pocetUzlu );
}

void uklidUkazatelu( unsigned **& dveDimenze, unsigned rozmer ) {
   if ( dveDimenze != NULL ) {
      for ( unsigned i = 0; i < rozmer; i++ ) {
         if ( dveDimenze[i] != NULL ) {
            delete [] dveDimenze[i];
            dveDimenze[i] = NULL;
         }
      }
      delete [] dveDimenze;
      dveDimenze = NULL;
   }
}

bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu ) {
   bool returnFlag = true;
   unsigned * vzdalenost, ** vzdalenostM, * predchudce, ** predchudceM;
   inicializaceNtoN( graf, pocetUzlu, vzdalenostM, predchudceM );
   unsigned idUzlu;
   cDijkstra * dijkstra;

   omp_set_num_threads( 5 );

#pragma omp parallel for private( idUzlu, vzdalenost, predchudce, dijkstra ) shared( vzdalenostM, predchudceM, returnFlag, pocetUzlu )
   for ( idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {

      dijkstra = new cDijkstra( idUzlu );
#ifdef DEBUG
      cout << "\nDijkstra pro uzel id = " << idUzlu << endl;      
#endif // DEBUG
      
      if ( dijkstra->spustVypocet( ) != true ) {
         cerr << "problem s vypoctem pro id = " << idUzlu<< endl;
         returnFlag = false;
      }
      else {
         // zkopiruje vysledek do matice
         vzdalenost = dijkstra->getVzdalenost( );
         predchudce = dijkstra->getPredchudce( );
         for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
            vzdalenostM[idUzlu][i] = vzdalenost[i];
            predchudceM[idUzlu][i] = predchudce[i];
         }
      }

      delete dijkstra;
   }

   cout << "Vzdalenosti:" << endl;
   vypisGrafu(cout, vzdalenostM, pocetUzlu);
   cout << endl << "Predchudci:" << endl;
   vypisGrafu(cout, predchudceM, pocetUzlu);

   uklidUkazatelu( predchudceM, pocetUzlu );
   uklidUkazatelu( vzdalenostM, pocetUzlu );

   return returnFlag;
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
   
   dijkstraNtoN( graf, pocetUzlu );

   uklid( graf, pocetUzlu );
   return MAIN_OK;
}

