/** funkceSpolecne.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 openMP paralelni implementace
 *                 Spolecne funkce pro nacitani / vypis dat
 *
 *
 */

#include "funkceSpolecne.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h> // getopts()

using namespace std;

void vypisUsage( ostream & os, const char * jmenoProgramu ) {
   os << "USAGE\n"
         "   " << jmenoProgramu << " [-t pocet_vlaken] -f vstupni_soubor\n"
         "   " << jmenoProgramu << " -h\n"
         "\n"
         "      vstupni_soubor   Soubor s daty grafu ve formatu ohodnocene incidencni\n"
         "                       matice (hrany ohodnocene vahami).\n"
         "                       Jedna se o n^2 + 1 unsigned hodnot, kde prvni hodnota\n"
         "                       udava cislo n (pocet uzlu, ruzny od nuly). Hodnota -\n"
         "                       urcuje neexistujici hranu\n"
         "\n"
         "      pocet_vlaken     Pocet vlaken pro paralelni cast vypoctu.\n"
         "                       Vychozi hodnota: 5\n"
         "\n"
         "      -h               Vypise tuto napovedu a skonci."
      << endl;
}

void uklid( unsigned ** graf, unsigned pocetUzlu ) {
   if ( graf != NULL ) {
      for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
         if ( graf[i] != NULL )
            delete [] graf[i];  
         graf[i] = NULL;
      }
      delete [] graf;
   }
   graf = NULL;
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
      
bool nactiPocetVlaken( const char * vstup, unsigned & pocetVlaken ) {
   istringstream iss( vstup );
   int tmp;
   iss >> tmp;
   if ( tmp < 1         ||
        iss.fail( )     ||
        zkontrolujPrazdnyVstup( iss ) != true 
      ) {
      return false;
   }
   pocetVlaken = (unsigned)tmp;
   return true;
}

bool zkontrolujSoubor( const char * optarg ) {
   ifstream f( optarg );
   bool navrat = true;
   if ( f.fail() ) 
      navrat = false;
   f.close();
   return navrat;
}

bool parsujArgumenty( int argc, char ** argv, unsigned & pocetVlaken, char *& souborSGrafem, unsigned & navrat ) {
   if ( argc < 2 ) {
      cerr << "Nedostatecny pocet argumentu" << endl;
      vypisUsage( cerr, argv[0] );
      navrat = MAIN_ERR_USAGE;
      return false;
   }

   int o = 0;
   souborSGrafem = NULL;
   opterr = 0; // zabrani getopt, aby vypisovala chyby
   // optstring '+' zajisti POSIX zadavani prepinacu, 
   //           ':' pro rozliseni neznameho prepinace od chybejiciho argumentu
   while ( ( o = getopt( argc, argv, "+:hf:t:" ) ) != -1 ) {
      switch ( o ) {
         case 'h':
            vypisUsage( cout, argv[0] );
            navrat = MAIN_OK;
            return false;

         case 'f':
            if ( zkontrolujSoubor( optarg ) != true ) {
               cerr << argv[0] << ": Soubor s daty neexistuje nebo neni citelny!" << endl;
               navrat = MAIN_ERR_VSTUP;
               return false;
            }
            souborSGrafem = optarg;
            break;

         case 't':
            if ( nactiPocetVlaken( optarg, pocetVlaken ) != true ) {
               cerr << argv[0] << ": Pocet vlaken musi byt kladne cele cislo" << endl;
               navrat = MAIN_ERR_VSTUP;
               return false;
            }
            break;

         case ':':
            cerr << argv[0] << ": Prepinac '-" << (char)optopt << "' vyzaduje argument." << endl;
            navrat = MAIN_ERR_VSTUP;
            return false;

         case '?':
            cerr << argv[0] << ": Neznamy prepinac '-" << (char)optopt << "'." << endl;
            navrat = MAIN_ERR_VSTUP;
            return false;

         default:
            navrat = MAIN_ERR_NEOCEKAVANA;
            return false;
      }
   }
   if ( souborSGrafem == NULL ) {
      cerr << argv[0] << ": Musi byt urceno jmeno souboru s grafem (prepinac '-f')" << endl;
      navrat = MAIN_ERR_VSTUP;
      return false;
   }
   if ( optind != argc ) {
      cerr << argv[0] << ": Chyba pri zpracovani parametru, nejsou dovoleny zadne argumenty." << endl;
      navrat = MAIN_ERR_VSTUP;
      return false;
   }

   return true;
}

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
      hodnota = UNSIGNED_NEKONECNO;
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

bool nactiGraf( istream & is, unsigned ** & graf, unsigned & pocetUzlu ) {
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

bool nactiData( const char * jmenoSouboru, unsigned ** & graf, unsigned & pocetUzlu ) {
   ifstream ifs( jmenoSouboru );
   if ( ifs.fail( ) ) {
      cerr << "nactiData(): Chyba! Nelze otevrit soubor '" << jmenoSouboru << "' pro cteni." << endl;
      return false;
   }
   bool ret = nactiGraf( ifs, graf, pocetUzlu );
   ifs.close( );
   return ret;
}

unsigned kontrolaGrafu( unsigned ** graf, unsigned pocetUzlu ) {
   unsigned ret = GRAF_NEORIENTOVANY;

   for ( unsigned i = 0 ; i < pocetUzlu - 1; i++ ) {
      if ( graf[i][i] != 0 ) {
         cerr << "Chyba formatu! Hodnota w(i,i) musi byt 0!"
                 "w(" << i << ',' << i << ") = " << graf[i][i] << endl;
         return GRAF_CHYBA;
      }

      for ( unsigned j = i + 1; j < pocetUzlu ; j++ ) {
         if ( graf[i][j] != graf[j][i] ) {
            ret = GRAF_ORIENTOVANY;
         }
      }
   }
   return ret;
}

void vypisGrafu( ostream & os, unsigned ** graf, unsigned pocetUzlu ) {
   os << setw(2) << ' ' << " |";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
     os << setw(2) << i << ' ';
   os << "\n---";
   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
      os << "---";
   

   for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
      os << '\n' << setw(2) << i << " |";
      for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
         if ( graf[i][j] == UNSIGNED_NEKONECNO )
            os << " - ";
         else
            os << setw(2) << graf[i][j] << ' ';
      }
   }
   os << endl;
}

