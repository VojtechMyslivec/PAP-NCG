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

#include "dijkstra.h"
#include "cDijkstra.h"
#include "funkceSpolecne.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <omp.h>

using namespace std;

void inicializaceNtoN( unsigned ** graf, unsigned pocetUzlu, 
                       unsigned **& vzdalenostM, unsigned **& predchudceM,
                       unsigned pocetVlaken
                     ) {
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
   
   // nastaveni poctu vlaken
   if ( pocetVlaken > pocetUzlu ) {
      cerr << "Varovani: pozadovany pocet vlaken (" << pocetVlaken 
           << ") je vetsi nez pocet uzlu (" << pocetUzlu 
           << "). Nastavuji na maximum (pocet uzlu)." << endl;
      pocetVlaken = pocetUzlu;
   }
#ifdef DEBUG
   cerr << "\nNastavuji pocet vlaken na " << pocetVlaken << endl;      
#endif // DEBUG
   omp_set_num_threads( pocetVlaken );

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

bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken ) {
   bool returnFlag = true;
   unsigned * vzdalenost, ** vzdalenostM, * predchudce, ** predchudceM;
   unsigned idUzlu;
   cDijkstra * dijkstra;

   inicializaceNtoN( graf, pocetUzlu, vzdalenostM, predchudceM, pocetVlaken );

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

   vypisVysledekMaticove( pocetUzlu, vzdalenostM, predchudceM );

   uklidUkazatelu( predchudceM, pocetUzlu );
   uklidUkazatelu( vzdalenostM, pocetUzlu );

   return returnFlag;
}

void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka, unsigned ** predchudce ) {
   cout << "Vzdalenosti:" << endl;
   vypisGrafu( cout, delka, pocetUzlu );
   cout << "\nPredchudci:" << endl;
   vypisGrafu( cout, predchudce, pocetUzlu );
}

