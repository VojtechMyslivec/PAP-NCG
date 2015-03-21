/** cDijkstra.h
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
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

#include "funkceSpolecne.h"
#include <climits>

#define DIJKSTRA_NEKONECNO    UNSIGNED_NEKONECNO
#define DIJKSTRA_NEDEFINOVANO UINT_MAX

// ============================================================================
// trida pro vypocet nejkratsich cest od jednoho uzlu ke vsem ostatnim
// jako vnitri pomocnou strukturu pouziva binarni haldu implementovanou polem

class cDijkstra {
   public:
      cDijkstra( unsigned ** graf, unsigned pocetUzlu );
      ~cDijkstra( );

      // Vypocte nejkratsi cesty od vsech uzlu ke vsem ostatnim uzlum
      // Fronta hran je implementovana prostym prohledavanim pole (uzly
      // jsou oznacene jako otevrene/zavrene, zda se ve fronte jeste 
      // vyskytuji ci nikoliv
      //
      // Slozitost vypoctu 1 -> n je O( |H|.|U| + |U|.|U| ) = 
      // = O( |U|^3 ). Celkova slozitost vypoctu N -> N je tedy O( |U|^4 )
      bool spustVypocet( );
      void vypisVysledekPoUzlech( unsigned uzelId ) const;
      void vypisVysledekMaticove( ) const;

   protected:
      // inicializace fronty, vzalenosti, predchudcu
      bool inicializace( unsigned idVychozihoUzlu );
      // Vypocte nejkratsi cesty od jednoho uzlu ke vsem ostatnim
      bool vypoctiProUzel( unsigned idVychozihoUzlu );

      // nalezne mimimum z fronty (neboli z otevrenych uzlu)
      // false v pripade prazdne fronty
      bool vyberMinimumZFronty( unsigned & idMinima ) const;
      void vypisFrontu( ) const;

      // ukazatel na matici vzdalenosti -- ohodnocenych hran
      // POZOR !! melka kopie !!
      unsigned ** graf;
      unsigned    pocetUzlu;

      unsigned * vzdalenost;
      unsigned * predchudce;
      // jestli je jiz cesta k uzlu vytvorena
      // zaroven urcije, jestli je jeste v prioritni fronte
      bool     * uzavreny;
      unsigned   pocetUzavrenychUzlu;
      
      // Pridane matice uchovavajici vsechny vysledky napric vsemi uzly
      unsigned** vzdalenostM;
      unsigned** predchudceM;
      unsigned idInstance;

};

#endif // CDIJKSTRA_kljnef29kjdsnf02

