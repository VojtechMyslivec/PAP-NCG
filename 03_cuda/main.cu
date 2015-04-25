/** main.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
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

#include <iostream>

#include "funkceSpolecne.h"
#ifdef FLOYDWARSHALL
   #include "floydWarshall.h"
#endif // FLOYDWARSHALL
#ifdef DIJKSTRA
   #include "dijkstra.h"
#endif // DIJKSTRA

using namespace std;

void mereni( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken ) {
   
//   double t1, t2;
//   t1 = omp_get_wtime( );
#ifdef DIJKSTRA
   dijkstraNtoN(  graf, pocetUzlu, pocetVlaken );
#endif // DIJKSTRA
#ifdef FLOYDWARSHALL
   floydWarshall( graf, pocetUzlu, pocetVlaken );
#endif // FLOYDWARSHALL
//   t2 = omp_get_wtime( );

#ifdef DEBUG
   cerr << " t1 = " << t1 << "; t2 = " << t2 << "; t = " << t2 - t1 << endl;
#endif // DEBUG

//   cerr << pocetUzlu << '	' << pocetVlaken << '	' << t2 - t1 << endl;
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

   mereni( graf, pocetUzlu, pocetVlaken );

   uklid( graf, pocetUzlu );
   
   return MAIN_OK;
}

