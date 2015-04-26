/** floydWarshall.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
 *                 funkce pro algoritmus Floyd-Warshall
 *
 *
 */

#include "floydWarshall.h"
#include <iomanip>
#include <omp.h>

void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken ) {
   unsigned ** delkaPredchozi = NULL;
   unsigned ** delkaAktualni  = NULL;
//   unsigned ** predchudcePredchozi = NULL;
//   unsigned ** predchudceAktualni  = NULL;
//   inicializace( pocetUzlu, graf, delkaPredchozi, delkaAktualni, predchudcePredchozi, predchudceAktualni, pocetVlaken );
   inicializace( pocetUzlu, graf, delkaPredchozi, delkaAktualni, pocetVlaken );

//   spustVypocet( pocetUzlu, graf, delkaPredchozi, delkaAktualni, predchudcePredchozi, predchudceAktualni );
   spustVypocet( pocetUzlu, graf, delkaPredchozi, delkaAktualni );

//   vypisVysledekMaticove( pocetUzlu, delkaAktualni, predchudceAktualni );
   vypisVysledekMaticove( pocetUzlu, delkaAktualni );

//   uklid( pocetUzlu, delkaPredchozi, delkaAktualni, predchudcePredchozi, predchudceAktualni );
   uklid( pocetUzlu, delkaPredchozi, delkaAktualni );
}

//void inicializace( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni, unsigned pocetVlaken ) {
void inicializace( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned pocetVlaken ) {
   delkaPredchozi      = new unsigned*[pocetUzlu];
   delkaAktualni       = new unsigned*[pocetUzlu];
//   predchudcePredchozi = new unsigned*[pocetUzlu];
//   predchudceAktualni  = new unsigned*[pocetUzlu];

   for ( unsigned i = 0; i < pocetUzlu; i++ ) {
      delkaPredchozi[i]      = new unsigned[pocetUzlu];
      delkaAktualni[i]       = new unsigned[pocetUzlu];
//      predchudcePredchozi[i] = new unsigned[pocetUzlu];
//      predchudceAktualni[i]  = new unsigned[pocetUzlu];

      for ( unsigned j = 0; j < pocetUzlu; j++ ) {
         delkaPredchozi[i][j] = graf[i][j];
//         if ( i == j || graf[i][j] == FW_NEKONECNO )
//            predchudcePredchozi[i][j] = FW_NEDEFINOVANO;
//         else
//            predchudcePredchozi[i][j] = i;
      }
   }
   
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

//void uklid( unsigned pocetUzlu, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni ) {
void uklid( unsigned pocetUzlu, unsigned **& delkaPredchozi, unsigned **& delkaAktualni ) {
   for ( unsigned i = 0; i < pocetUzlu; i++ ) {
      delete [] delkaPredchozi[i];
      delete [] delkaAktualni[i];
//      delete [] predchudcePredchozi[i];
//      delete [] predchudceAktualni[i];
   }
   delete [] delkaPredchozi;
   delete [] delkaAktualni;
//   delete [] predchudcePredchozi;
//   delete [] predchudceAktualni;

   delkaPredchozi      = delkaAktualni      = NULL;
//   predchudcePredchozi = predchudceAktualni = NULL;
}

//void spustVypocet( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni ) {
void spustVypocet( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni ) {
   unsigned novaVzdalenost;
   
   unsigned i, j, k;
   i = j = k = 0;
//#pragma omp parallel private( i, j, novaVzdalenost ) shared( k, delkaPredchozi, delkaAktualni, predchudcePredchozi, predchudceAktualni )
#pragma omp parallel private( i, j, novaVzdalenost ) shared( k, delkaPredchozi, delkaAktualni )
   while ( k < pocetUzlu ) {
#pragma omp for
      for ( i = 0; i < pocetUzlu; i++ ) {
         for ( j = 0; j < pocetUzlu; j++ ) {
            // osetreni nekonecna
            if ( delkaPredchozi[i][k] == FW_NEKONECNO || delkaPredchozi[k][j] == FW_NEKONECNO )
                novaVzdalenost = FW_NEKONECNO;
            else
                novaVzdalenost = delkaPredchozi[i][k] + delkaPredchozi[k][j];

            // pokud nalezne kratsi cestu, zapise ji a zmeni predchudcePredchozi
            if ( novaVzdalenost < delkaPredchozi[i][j] ) {
                delkaAktualni[i][j]      = novaVzdalenost;
//                predchudceAktualni[i][j] = predchudcePredchozi[k][j];
            }
            // jinak delka i predchudce zustavaji
            else {
                delkaAktualni[i][j]      = delkaPredchozi[i][j];
//                predchudceAktualni[i][j] = predchudcePredchozi[i][j];
            }
         }
      }
#pragma omp single
      {
         // prohozeni predchozi a aktualni
         prohodUkazatele( delkaPredchozi,      delkaAktualni );
//         prohodUkazatele( predchudcePredchozi, predchudceAktualni );
         k++;
      }
   }

   // prohozeni predchozi a aktualni, aby vysledky byly v aktualnim ( po skonceni cyklu jsou vysledky v predchozim )
   prohodUkazatele( delkaPredchozi,      delkaAktualni );
//   prohodUkazatele( predchudcePredchozi, predchudceAktualni );
}

void prohodUkazatele( unsigned **& ukazatel1, unsigned **& ukazatel2 ) {
   unsigned ** pomocny;
   pomocny   = ukazatel1;
   ukazatel1 = ukazatel2;
   ukazatel2 = pomocny;
}

//void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka, unsigned ** predchudce ) {
void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka ) {
   vypisGrafu( cout, delka, pocetUzlu );
//   cout << endl;
//   vypisGrafu( cout, predchudce, pocetUzlu );
}

