/** cDijkstra.hpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 openMP paralelni implementace
 *                 trida cDijkstra pro Dijkstruv algoritmus
 *                    upraven vypocet prioritni fronty -- misto haldy for cylky, 
 *                    aby slo paralelizovat
 *
 *
 */

#ifndef CDIJKSTRA_kljnef29kjdsnf02
#define CDIJKSTRA_kljnef29kjdsnf02

#ifdef DEBUG2
   #define DEBUG
#endif // DEBUG2

#include "funkceSpolecne.hpp"
#include <climits>

#define DIJKSTRA_NEKONECNO    UNSIGNED_NEKONECNO
#define DIJKSTRA_NEDEFINOVANO UINT_MAX

// ============================================================================
// trida pro vypocet nejkratsich cest od jednoho uzlu ke vsem ostatnim
// jako vnitri pomocnou strukturu pouziva binarni haldu implementovanou polem

class cDijkstra {
   public:
      cDijkstra( unsigned idVychozihoUzlu );
      static void inicializace( unsigned ** graf, unsigned pocetUzlu );
      ~cDijkstra( );

      // Vypocte nejkratsi cesty od daneho uzlu ke vsem ostatnim uzlum
      // Fronta hran je implementovana prostym prohledavanim pole (uzly
      // jsou oznacene jako otevrene/zavrene, zda se ve fronte jeste 
      // vyskytuji ci nikoliv
      //
      // Slozitost vypoctu 1 -> n je O( |H|.|U| + |U|.|U| ) = 
      // = O( |U|^3 ).
      bool spustVypocet( );
      
      // navrat poli s vysledky
//      unsigned * getPredchudce( ) const;
      unsigned * getVzdalenost( ) const;

      void vypisVysledekPoUzlech( ) const;

   protected:
      // nalezne mimimum z fronty (neboli z otevrenych uzlu)
      // false v pripade prazdne fronty
      bool vyberMinimumZFronty( unsigned & idMinima ) const;
      void vypisFrontu( ) const;

      // ukazatel na matici vzdalenosti -- ohodnocenych hran
      // POZOR !! melka kopie !!
      static unsigned ** graf;
      static unsigned    pocetUzlu;

      unsigned idVychozihoUzlu;

      unsigned * vzdalenost;
//      unsigned * predchudce;
      // jestli je jiz cesta k uzlu vytvorena
      // zaroven urcije, jestli je jeste v prioritni fronte
      bool     * uzavreny;
      unsigned   pocetUzavrenychUzlu;
};

#endif // CDIJKSTRA_kljnef29kjdsnf02

