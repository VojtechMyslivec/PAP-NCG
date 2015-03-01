/** dijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 algoritmus Dijkstra
 *                 sekvencni cast
 *
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <climits>

#define MAIN_OK            0
#define MAIN_ERR_VSTUP     1
#define MAIN_ERR_ORIENTACE 2

#define PARAMETER_HELP1 "--help"
#define PARAMETER_HELP2 "-h"

#define NACTI_OK           0
#define NACTI_NEKONECNO    1
#define NACTI_ERR_PRAZDNO  2
#define NACTI_ERR_ZNAMENKO 3
#define NACTI_ERR_CISLO    4
#define NACTI_ERR_TEXT     5

#define GRAF_ORIENTOVANY    true
#define GRAF_NEORIENTOVANY  false

#define DIJKSTRA_NEKONECNO    UINT_MAX
#define DIJKSTRA_NEDEFINOVANO UINT_MAX

// makra na vypocet synu / otce v uplnem bin. strome implementovanym pomoci
// pole. Indexy jsou samozrejme od nuly
#define otec(i)  ((i)-1)/2
#define levy(i)  (2*(i))+1
#define pravy(i) (2*(i))+2

using namespace std;

// globalni promenne ==========================================================
unsigned ** graf = NULL;
unsigned pocetUzlu = 0;


// ============================================================================
// deklarace tridy cDijkstra ==================================================
// trida pro vypocet nejkratsich cest od jednoho uzlu ke vsem ostatnim
// jako vnitri pomocnou strukturu pouziva binarni haldu implementovanou polem

class cDijkstra {
   public:
      cDijkstra( );
      ~cDijkstra( );

      bool spustVypocet( unsigned idVychozihoUzlu );
//      unsigned * getPredchudci( ) const;
//      unsigned * getVzdalenosti( ) const;
      void vypisVysledek( ) const;

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

// ============================================================================
// definice tridy cDijkstra ===================================================
cDijkstra::cDijkstra( ) {
   halda         = new unsigned[pocetUzlu];
   indexyVHalde  = new unsigned[pocetUzlu];
   vzdalenost    = new unsigned[pocetUzlu];
   predchudce    = new unsigned[pocetUzlu];
   uzavreny      = new bool[pocetUzlu];
   velikostHaldy = 0;
}

cDijkstra::~cDijkstra( ) {
   if ( halda != NULL ) 
      delete [] halda;
   if ( indexyVHalde != NULL ) 
      delete [] indexyVHalde;
   if ( vzdalenost != NULL ) 
      delete [] vzdalenost;
   if ( predchudce != NULL ) 
      delete [] predchudce;
   if ( uzavreny   != NULL ) 
      delete [] uzavreny;
}

bool cDijkstra::spustVypocet( unsigned idVychozihoUzlu ) {
   unsigned idUzlu;
   unsigned vzdalenostUzlu, vzdalenostSouseda, vzdalenostHrany, novaVzdalenost;
       
   // inicializace haldy, vzalenosti, predchudcu
   if ( inicializace( idVychozihoUzlu ) != true ) {
      return false;
   }


//vypisHaldu( );
   // dokud nenavstivi vsechny uzly
   while ( jePrazdnaHalda( ) != true ) {
      // uzly navstevuje podle nejmensi vzdalenosti
      vemMinimumZHaldy( idUzlu );
      uzavreny[idUzlu] = true;
//cout << "\nzpracovavam uzel " << idUzlu << endl;
//vypisHaldu( );

      vzdalenostUzlu = vzdalenost[idUzlu];
      // pokud je stale nekonecna, znamena to nedostupny uzel;
      if ( vzdalenostUzlu >= DIJKSTRA_NEKONECNO ) {
//cout << "preskakuji " << idUzlu << " vzd.: " << vzdalenostUzlu << endl;
         continue;
      }

      // pro vsechny sousedy, tedy for cyklus pres matici vzdalenosti
      for ( unsigned idSouseda = 0 ; idSouseda < pocetUzlu ; idSouseda++ ) {
         // pokud je uzavreny, preskocim
         if ( uzavreny[idSouseda] == true ) {
            continue;
         }

         vzdalenostSouseda = vzdalenost[idSouseda];
         vzdalenostHrany   = graf[idUzlu][idSouseda];
         // kontrola, aby nepretekla hodnota
         if ( vzdalenostHrany >= DIJKSTRA_NEKONECNO ) 
            novaVzdalenost = DIJKSTRA_NEKONECNO;
         else
            novaVzdalenost = vzdalenostUzlu + vzdalenostHrany;
         
         // nalezeni kratsi vzdalenosti
         if ( novaVzdalenost < vzdalenostSouseda ) {
//cout << "   nova vzdalenost z " << idUzlu << " do " << idSouseda << " = " << vzdalenostHrany << '(' << novaVzdalenost << ')'<< endl;
            predchudce[idSouseda] = idUzlu;
            vzdalenost[idSouseda] = novaVzdalenost;

            // oprava haldy dle nove hodnoty
            opravPoziciVHalde( indexyVHalde[idSouseda] );
//vypisHaldu( );
         }
      }
   }

   return true;
}


bool cDijkstra::inicializace( unsigned idVychozihoUzlu ) {
   if ( idVychozihoUzlu >= pocetUzlu ) {
      cerr << "inicializace(): Chyba! id uzlu je vyssi nez pocet uzlu.";
      return false;
   }

   velikostHaldy = 0;
   for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
      vzdalenost[idUzlu] = DIJKSTRA_NEKONECNO;
      predchudce[idUzlu] = DIJKSTRA_NEDEFINOVANO;
      uzavreny[idUzlu]   = false;
      pridejPrvekDoHaldy( idUzlu );
   }
   vzdalenost[idVychozihoUzlu] = 0;

   return true;
}

void cDijkstra::opravPoziciVHalde( unsigned pozice ) {
   // nalezne misto, kam ma uzel na pozici patrit
   // indexDoHaldy ukazuje na opravovane misto, indexOtceVHalde na otce tohoto mista
   // idUzluOtce je hodnota v halde na pozici otce a hodnotaUzlu je vzdalenost uzlu na pozici
   unsigned idUzlu      = halda[pozice];
   unsigned hodnotaUzlu = vzdalenost[idUzlu];
//cout << " + oprav pozici " << pozice << "; id uzlu " << idUzlu << "; hodnota Uzlu " << hodnotaUzlu << endl;

   unsigned indexDoHaldy, indexOtceVHalde, idUzluOtce;
   for ( indexDoHaldy = pozice ; indexDoHaldy > 0 ; indexDoHaldy = otec(indexDoHaldy) ) {
      indexOtceVHalde = otec(indexDoHaldy);
      idUzluOtce      = halda[indexOtceVHalde];
//cout << " ++ indexOtceVHalde " << indexOtceVHalde << "; idUzluOtce " << idUzluOtce << "; vzdalenost otce " << vzdalenost[idUzluOtce] << endl;

      // pokud je vzdalenost otce mensi nez hodnota uzlu, skonci a indexDoHaldy ukazuje 
      // na spravnou pozici
      if ( vzdalenost[idUzluOtce] <= hodnotaUzlu )
         break;

      // presun otce indexu na nizsi pozici
      halda[indexDoHaldy]      = idUzluOtce;
      indexyVHalde[idUzluOtce] = indexDoHaldy;
   }

   // na nalezene (uvolnene) misto ulozi id puvodniho uzlu
   halda[indexDoHaldy]  = idUzlu;
   indexyVHalde[idUzlu] = indexDoHaldy;

}

bool cDijkstra::pridejPrvekDoHaldy( unsigned idUzlu ) {
   if ( velikostHaldy >= pocetUzlu ) {
      cerr << "vemMinimumZHaldy(): Chyba! Preteceni haldy!" << endl;
      return false;
   }
   
   // prida idUzlu na prvni volne misto v halde, opravi haldu a zvysi pocet prvku v halde
   halda[velikostHaldy] = idUzlu;
   indexyVHalde[idUzlu] = velikostHaldy;
   opravPoziciVHalde( velikostHaldy );
   velikostHaldy++;

   return true;
}

void cDijkstra::haldujRekurzivne( unsigned indexOtce ) {
   unsigned indexLeveho       =  levy( indexOtce );
   unsigned indexPraveho      = pravy( indexOtce );

   unsigned vzdalenostOtce    = vzdalenost[halda[indexOtce]];
//cout << " +       indexOtce " << indexOtce      << ";      indexLeveho " << indexLeveho      << ";      indexPraveho " << indexPraveho << endl;

   // nejmensi ze tri
   unsigned indexNejmensiho, vzdalenostNejmensiho, pomocny;
   if (  indexLeveho < velikostHaldy    &&
         vzdalenost[halda[indexLeveho]] < vzdalenostOtce )
      indexNejmensiho = indexLeveho;
   else 
      indexNejmensiho = indexOtce;
   vzdalenostNejmensiho = vzdalenost[indexNejmensiho]; 

   if ( indexPraveho < velikostHaldy    &&
        vzdalenost[halda[indexPraveho]] < vzdalenostNejmensiho )
      indexNejmensiho = indexPraveho;
   vzdalenostNejmensiho = vzdalenost[indexNejmensiho]; 

//cout << " ++ vzdalenostOtce " << vzdalenostOtce << "; indexNejmensiho " << indexNejmensiho << "; vzdalenostNejmensiho " << vzdalenostNejmensiho << endl;

   // pokud je pod otcem mensi prvek, prohod je a zavolej se rekurzivne
   if ( indexNejmensiho != indexOtce ) {
      pomocny                 = halda[indexOtce];
      halda[indexOtce]        = halda[indexNejmensiho];
      halda[indexNejmensiho]  = pomocny;

      indexyVHalde[halda[indexOtce]]        = indexOtce;
      indexyVHalde[halda[indexNejmensiho]]  = indexNejmensiho;

      // prvek byl presunut na pozici nejmensi (pravy ci levy syn), halduj tam
      haldujRekurzivne( indexNejmensiho );
   }
   // jinak je otec na spravnem miste
}

bool cDijkstra::vemMinimumZHaldy( unsigned & idUzlu ) {
   if ( velikostHaldy == 0 ) {
      cerr << "vemMinimumZHaldy(): Chyba! Podteceni haldy!" << endl;
      return false;
   }
   // minimum je na vrchu haldy
   idUzlu   = halda[0];
   indexyVHalde[idUzlu] = DIJKSTRA_NEDEFINOVANO;
   // posledni prvek ulozi na vrchol, zmensi pocet prvku a opravi haldu
   velikostHaldy--;
   halda[0] = halda[velikostHaldy];
   indexyVHalde[halda[0]] = 0;

   haldujRekurzivne( 0 );

   return true;
}

bool cDijkstra::jePrazdnaHalda( ) const {
   return velikostHaldy == 0;
}

void cDijkstra::vypisVysledek( ) const {
   unsigned hodnota;
   cout << "id uzlu:      ";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
         cout << setw(2) << i << " ";
   }
   cout << "\n"
           "Vzdalenosti:  ";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      hodnota = vzdalenost[i];
      if ( hodnota == DIJKSTRA_NEKONECNO )
         cout << " - ";
      else
         cout << setw(2) << hodnota << " ";
   }
   cout << "\n"
           "Predchudci:   ";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      hodnota = predchudce[i];
      if ( hodnota == DIJKSTRA_NEDEFINOVANO )
         cout << " - ";
      else
         cout << setw(2) << hodnota << " ";
   }
   cout << endl;
}

void cDijkstra::vypisHaldu( ) const {
   cout << "Halda: \n";
   unsigned pravy = 0;
   for ( unsigned i = 0 ; i < velikostHaldy ; i++ ) {
      if ( i > pravy ) {
         cout << '\n';
         pravy = pravy(pravy);
      }
      cout << halda[i] << '(' << vzdalenost[halda[i]] << ") ";
      if ( i % 2 == 0 ) 
         cout << ' ';
   }
   cout << endl;
}

// funkce programu ============================================================
void vypisUsage( const char * jmenoProgramu ) {
   cerr << "USAGE\n"
           "   " << jmenoProgramu << " vstupni_soubor\n"
           "   " << jmenoProgramu << " -h | --help\n"
           "\n"
           "      vstupni_soubor   Soubor s daty grafu ve formatu ohodnocene incidencni\n"
           "                       matice (hrany ohodnocene vahami).\n"
           "                       Jedna se o n^2 + 1 unsigned hodnot, kde prvni hodnota\n"
           "                       udava cislo n (pocet uzlu, ruzny od nuly).\n"
           "\n"
           "      -h, --help       Vypise tuto napovedu a skonci."
        << endl;
}

void uklid( ) {
   if ( graf != NULL ) {
      for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
         if ( graf[i] != NULL )
            delete [] graf[i];  
      }
      delete [] graf;
   }
}

bool zkontrolujPrazdnyVstup( istream & is ) {
   char c = '\0';
   while ( true ) {
      c = is.get();
      if ( is.fail() ) 
         return true;
      if ( ! ( c == ' ' || c == '\t' || c == '\n' || c == '\r' ) )
         return false;
   }
   return true;
}

// nacte jednu unsigned hodnotu ze vstupu
// pokud misto unsigned cisla nalezne - (nasledovanou prazdnym znakem)
// ulozi do hodnoty DIJKSTRA_NEKONECNO
unsigned nactiHodnotu( istream & is, unsigned & hodnota ) {
   char c;
   // sebrani prazdnych znaku
   do {
      is.get( c );
      // pokud je prazdny vstup, skonci
      if ( is.fail() ) {
         return NACTI_ERR_PRAZDNO;
      }
   } while ( c == ' ' || c == '\t' || c == '\n' );

   // na vstupu je - 
   if ( c == '-' ) {
      // zkontroluje, jestli nasleduje prazdny znak   
      is.get( c );
      if ( c != ' ' && c != '\t' && c != '\n' ) {
         // pokud nasledovalo cislo, byl to pokus o zadani zaporne hodnoty
         if ( c >= '0' && c <= '9') {
            return NACTI_ERR_ZNAMENKO;
         }
         // jinak jde o nejaky spatny vstup (text)
         return NACTI_ERR_TEXT;
      }

      // kolem znaku - jsou spravne prazdne znaky, reprezentuje tedy nekonecno
      hodnota = DIJKSTRA_NEKONECNO;
      return NACTI_NEKONECNO;
   }
   // jinak to ma byt cislo

   // vrati znak na vstup a pokusi se nacist cislo
   is.putback( c );
   is >> hodnota;
   // pokud se nepodarilo nacist, nebylo na vstupu cislo
   if ( is.fail( ) ) {
      return NACTI_ERR_CISLO;
   }
   return NACTI_OK;
}

bool nactiGraf( istream & is ) { // , unsigned & ** graf, unsigned & pocetUzlu ) {
   is >> pocetUzlu;
   if ( is.fail( ) || pocetUzlu < 1 ) {
      cerr << "nactiGraf(): Chyba! Pocet uzlu musi byt kladne cislo" << endl;
      return false;
   }

   graf = new unsigned*[pocetUzlu];
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      graf[i] = new unsigned[pocetUzlu];
   }

   unsigned ret;
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
         ret = nactiHodnotu( is, graf[i][j] );
         // pouze se rozhoduje vypis pripadne chybove hlasky
         switch ( ret ) {
            case NACTI_OK:
            case NACTI_NEKONECNO:
               continue;
               break;
            case NACTI_ERR_PRAZDNO:
               cerr << "nactiGraf(): Chyba! V ocekavam n^2 nezapornych hodnot (n = " << pocetUzlu << ")." << endl;
               return false;
               break;
            case NACTI_ERR_ZNAMENKO:
            case NACTI_ERR_CISLO:
            case NACTI_ERR_TEXT:
               cerr << "nactiGraf(): Chyba! Ocekavam pouze nezaporne hodnoty nebo '-' pro nekonecno." << endl;
               return false;
            default:
               cerr << "nactiGraf(): Neocekavana chyba vstupu." << endl;
               return false;
               break;
         }
      }
   }

   if ( zkontrolujPrazdnyVstup( is ) != true ) {
      cerr << "nactiGraf(): Chyba! Na vstupu je vice dat, nez je ocekavano (n = " << pocetUzlu << ")." << endl;
      return false;
   }
   return true;
}

bool nactiData( char * jmenoSouboru ) {
   ifstream ifs( jmenoSouboru );
   if ( ifs.fail( ) ) {
      cerr << "nactiData(): Chyba! Nelze otevrit soubor '" << jmenoSouboru << "' pro cteni." << endl;
      return false;
   }
   return nactiGraf( ifs );
}

bool kontrolaGrafu( ) { // const unsigned ** graf, unsigned pocetUzlu ) {
   for ( unsigned i = 0 ; i < pocetUzlu - 1; i++ ) {
      for ( unsigned j = i + 1; j < pocetUzlu ; j++ ) {
         if ( graf[i][j] != graf[j][i] ) {
            return GRAF_ORIENTOVANY;
         }
      }
   }
   return GRAF_NEORIENTOVANY;
}

void vypisGrafu( ) { // const unsigned ** graf, unsigned pocetUzlu ) {
   cout << setw(2) << ' ' << " |";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
      cout << setw(2) << i << ' ';
   cout << "\n---";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
      cout << "---";
   

   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      cout << '\n' << setw(2) << i << " |";
      for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
         if ( graf[i][j] == DIJKSTRA_NEKONECNO )
            cout << " - ";
         else
            cout << setw(2) << graf[i][j] << ' ';
      }
   }
   cout << endl;
}



// main =======================================================================
int main( int argc, char ** argv ) {
//   cout << "Hello Dijkstra" << endl;

   if ( argc != 2 ) {
      vypisUsage( argv[0] );
      return MAIN_ERR_VSTUP;
   }
   if ( 
         strncmp( argv[1], PARAMETER_HELP1, sizeof(PARAMETER_HELP1) ) == 0 || 
         strncmp( argv[1], PARAMETER_HELP2, sizeof(PARAMETER_HELP2) ) == 0 
      ) {
      vypisUsage( argv[0] );
      return MAIN_OK;
   }

   if ( nactiData( argv[1] ) != true ) {
      uklid( );
      return MAIN_ERR_VSTUP;
   }

   vypisGrafu( );
   if ( kontrolaGrafu( ) != GRAF_NEORIENTOVANY ) {
      cerr << "Graf je orientovany" << endl;
   }
   else {
      cerr << "Graf je neorientovany" << endl;
   }

   cerr << endl;

   cDijkstra * dijkstra = new cDijkstra( );
   dijkstra->spustVypocet( 0 );
   dijkstra->vypisVysledek( );

   delete dijkstra;

   uklid( );
   return MAIN_OK;
}

