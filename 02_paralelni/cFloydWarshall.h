/** cFloydWarshall.h
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 trida cFloydWarshall pro algoritmus Floyd-Warshall
 *
 *
 */

#ifndef CFLOYDWARSHALL_kljnef29kjdsnf02
#define CFLOYDWARSHALL_kljnef29kjdsnf02

#ifdef DEBUG2
   #define DEBUG
#endif // DEBUG2

#include "funkceSpolecne.h"
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

class cFloydWarshall {
   public:
      //inicializuje a nastavi vnitrni promenne tridy jako matice pocetUlzu^2
      cFloydWarshall(unsigned ** graf, unsigned pocetUzlu);
      virtual ~cFloydWarshall();

      // vypocte nejkratsi cesty od vsech uzlu ke vsem ostatnim
      // slozitost vypoctu je O( |U|^3 )
      // (cesty od vsech uzlu do kazdheo uzlu pres jakykoliv uzel)
      void spustVypocet();

      // prohodi dva pointery mezi sebou pres pomocny pointer
      // slozitost je O(1)
      void prohodPredchoziAAktualni();

      // vypise vysledek po startovnich uzlech ve tvaru
      // id uzlu  X, uzly, jejich vzdalenosti a predchudci
      void vypisVysledekPoUzlech() const;

      // vypise vysledek jako matici vzdalenosti a matici predchudcu
      void vypisVysledekMaticove() const;

   protected:
      unsigned ** delkaPredchozi;
      unsigned ** delkaAktualni;
      unsigned ** predchudcePredchozi;
      unsigned ** predchudceAktualni;
      unsigned    pocetUzlu;

};

#endif // CFLOYDWARSHALL_kljnef29kjdsnf02

