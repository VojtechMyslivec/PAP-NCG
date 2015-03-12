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
 *                 trida cDijkstra pro Dijkstruc algoritmus
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

// makra na vypocet synu / otce v uplnem bin. strome implementovanym pomoci
// pole. Indexy jsou samozrejme od nuly
#define otec(i)  ((i)-1)/2
#define levy(i)  (2*(i))+1
#define pravy(i) (2*(i))+2

// ============================================================================
// trida pro vypocet nejkratsich cest od jednoho uzlu ke vsem ostatnim
// jako vnitri pomocnou strukturu pouziva binarni haldu implementovanou polem

class cDijkstra {
   public:
      cDijkstra( unsigned ** graf, unsigned pocetUzlu );
      ~cDijkstra( );

      // vypocte nejkratsi cesty od uzlu idVychozihoUzlu ke vsem ostatnim
      // Fronta hran je implementovana binarni haldou
      // slozitost vypoctu je tedy O( |H|.log(|U|) + |U|.log(|U|) ) = 
      // = O( |U|^2 . log(|U|) )
      unsigned* spustVypocet( unsigned idVychozihoUzlu );
//      unsigned * getPredchudci( ) const;
//      unsigned * getVzdalenosti( ) const;
      void vypisVysledekPoUzlech( unsigned uzelId ) const;
      void vypisVysledekMaticove( ) const;
      void vypisHaldu( ) const;

   protected:
      bool inicializace( unsigned idVychozihoUzlu );

      // prida prvek do haldy
      // false v pripade plne haldy
      bool pridejPrvekDoHaldy( unsigned idUzlu );
      // opravi pozici v halde od daneho indexu do haldy
      // (smerem nahoru, hodi se pri vkladani ci prvku zmenseni vzdalenosti)
      void opravPoziciVHalde( unsigned pozice );
      // vezme prvek z haldy (uzel s dosavadni nejkratsi cestou)
      // pri zachovani vlastnosti haldy
      // false v pripade prazdne haldy
      bool vemMinimumZHaldy( unsigned & idUzlu );
      // vraci true, false pokud je, neni prazdna halda
      bool jePrazdnaHalda( ) const;
      // zkontroluje a opravi haldu od indexu indexOtce rekurzivne
      // do podstromu
      // heapify -- synove Otce musi byt bin. haldy
      void haldujRekurzivne( unsigned indexOtce );
      // nastavi novou vzdalenost uzlu a opravi jeho pozici v halde
      // nutne volat tuto metodu, aby se nezapomnelo opravit haldu!!!!!!!
      // Tato metoda jiz nekontroluje, zda je idUzlu platne. Musi se osetrit 
      // pred volanim!!!
      void nastavVzdalenostUzlu( unsigned idUzlu, unsigned novaVzdalenost );

      // ukazatel na matici vzdalenosti -- ohodnocenych hran
      // TODO !! melka kopie !!
      unsigned ** graf;
      unsigned    pocetUzlu;

      // Binarni halda implementovana polem. Pro otce, leveho a praveho syna 
      // existuji makra otec(i), levy(i) a pravy(i), kde i je index v poli.
      // V poli jsou ulozene indexy uzlu a jsou serazene podle jejich vzdalenosti
      unsigned * halda;
      unsigned   velikostHaldy;
      // pocet uzlu grafu je zaroven maximalni pocet prvku v halde
//      unsigned   pocetUzlu;
      // "halda ^ -1" pole, ktere je indexovano jako uzly, kde hodnoty jsou indexy v halde
      unsigned * indexyVHalde;
      unsigned * vzdalenost;
      unsigned * predchudce;
      // odpovida hodnote (neni v halde)
      bool     * uzavreny;
};

#endif // CDIJKSTRA_kljnef29kjdsnf02

