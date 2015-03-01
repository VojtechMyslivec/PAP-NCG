/** funkceSpolecne.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 Spolecne funkce pro nacitani / vypis dat
 *
 *
 */

#include "funkceSpolecne.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

// funkce programu ============================================================
void vypisUsage( ostream & os, const char * jmenoProgramu ) {
   os << "USAGE\n"
         "   " << jmenoProgramu << " vstupni_soubor\n"
         "   " << jmenoProgramu << " " PARAMETER_HELP1 " | " PARAMETER_HELP2 "\n"
         "\n"
         "      vstupni_soubor   Soubor s daty grafu ve formatu ohodnocene incidencni\n"
         "                       matice (hrany ohodnocene vahami).\n"
         "                       Jedna se o n^2 + 1 unsigned hodnot, kde prvni hodnota\n"
         "                       udava cislo n (pocet uzlu, ruzny od nuly).\n"
         "\n"
         "      " PARAMETER_HELP1 ", " PARAMETER_HELP2  "       Vypise tuto napovedu a skonci."
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

bool nactiData( char * jmenoSouboru, unsigned ** & graf, unsigned & pocetUzlu ) {
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

