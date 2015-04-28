/** main.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 openMP paralelni implementace
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

#include "funkceSpolecne.hpp"
#ifdef FLOYDWARSHALL
   #include "floydWarshall.hpp"
#endif // FLOYDWARSHALL
#ifdef DIJKSTRA
   #include "dijkstra.hpp"
#endif // DIJKSTRA

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

using namespace std;

void mereni( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken ) {
   
#ifdef MERENI
   double t1, t2;
   t1 = omp_get_wtime( );
#endif // MERENI

#ifdef DIJKSTRA
   dijkstraNtoN(  graf, pocetUzlu, pocetVlaken );
#endif // DIJKSTRA
#ifdef FLOYDWARSHALL
   floydWarshall( graf, pocetUzlu, pocetVlaken );
#endif // FLOYDWARSHALL

#ifdef MERENI
   t2 = omp_get_wtime( );
   cerr << pocetUzlu << '	' << pocetVlaken << '	' << t2 - t1 << endl;
#endif // MERENI
}

// main =======================================================================
int main( int argc, char ** argv ) {
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

   mereni( graf, pocetUzlu, pocetVlaken );

   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

