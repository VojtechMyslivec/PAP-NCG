/** funkceSpolecne.hpp
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

#ifndef FUNKCE_SPOLECNE_aohiuefijn39nvkjns92
#define FUNKCE_SPOLECNE_aohiuefijn39nvkjns92

#include <iostream>
#include <climits>

#define UNSIGNED_NEKONECNO  UINT_MAX

#define MAIN_OK            0
#define MAIN_ERR_USAGE     1
#define MAIN_ERR_VSTUP     2
#define MAIN_ERR_GRAF      3
#define MAIN_ERR_NEOCEKAVANA 10

#define NACTI_OK           0
#define NACTI_NEKONECNO    1
#define NACTI_ERR_PRAZDNO  2
#define NACTI_ERR_ZNAMENKO 3
#define NACTI_ERR_CISLO    4
#define NACTI_ERR_TEXT     5

#define GRAF_ORIENTOVANY   0
#define GRAF_NEORIENTOVANY 1
#define GRAF_CHYBA         2

using namespace std;

// vypise usage na vystupni stream os, vola se s argumentem argv[0] 
// jako jmenem programu
void vypisUsage( ostream & os, const char * jmenoProgramu );

// smaze alokovany graf (provadi kontrolu ukazatele na NULL) a nastavi
// ukazatel(e) na NULL
void uklid( unsigned ** graf, unsigned pocetUzlu );

// zkonstroluje, zda vstupni stream is je uz prazdny (muze obsahovat prazdne znaky)
//
//   true   stream je prazdny
//   false  stream neni/nebyl prazdny
bool zkontrolujPrazdnyVstup( istream & is );

// Funkce zajisti nacteni a kontrolu parametru.
// Do vystupnich promenych uklada:
//    pocetVlaken      pocet pozadovanych vlaken       -- prepinac -t
//    souborSGrafem    jmeno souboru se vstupnimi daty -- prepinac -f (povinny!)
//    navrat           navratovy kod v pripade selhani / chyby
//                     tedy  v pripade vraceni false
//
// Navratove hodnoty
//    false  Chyba vstupu -- prepinacu. Danou chybu vrati na stderr 
//           a ulozu navratovou hodnotu do parametru navrat.
//           Prepinac -h prinuti funkci skoncit s chybou, ale navratova
//           hodnota bude MAIN_OK.
//    true   Vse v poradku, parametry byly uspesne nacteny
bool parsujArgumenty( int argc, char ** argv, unsigned & pocetVlaken, char *& souborSGrafem, unsigned & navrat );

// nacte jednu unsigned hodnotu ze vstupu
// pokud misto unsigned cisla nalezne - (nasledovanou prazdnym znakem)
// ulozi do hodnoty DIJKSTRA_NEKONECNO
//
//   NACTI_OK           v poradku se podarilo nacist jednu unsigned hodnotu
//   NACTI_NEKONECNO    na vstupu byl znak '-' oddeleny mezerami, do hodnoty byla
//                      prirazena hodnota UNSIGNED_NEKONECNO
//   NACTI_ERR_TEXT     na vstupu byl nejaky retezec zacinajici znakem -
//   NACTI_ERR_PRAZDNO  na vstupu jiz nebyl zadny platny znak
//   NACTI_ERR_ZNAMENKO na vstupu byla zaporna hodnota
//   NACTI_ERR_CISLO    na vstupu nebylo cislo
unsigned nactiHodnotu( istream & is, unsigned & hodnota );

// funkce, ktera ze vstupniho streamu is nacte graf ve formatu 
// n w(i,j), kde n je pocet uzlu (unsigned n > 0) a nasleduje 
// n^2 unsigned hodnot, reprezentujici matici vah hran z uzlu i do j
// vaha muze byt nahrazen znakem '-' (ohranicenym prazdnymi znaky),
// ktery reprezentuje nekonecno.
//
//   true   uspesne nacteni grafu
//   false  chyba vstupu
bool nactiGraf( istream & is, unsigned ** & graf, unsigned & pocetUzlu );

// funkce otevre soubor s nazvem v c stringu a z neho se pokusi nacist graf
//
//   true   uspesne nacteni dat ze souboru
//   false  chyba souboru ci chyba vstupu
bool nactiData( const char * jmenoSouboru, unsigned ** & graf, unsigned & pocetUzlu );

// funkce, ktera zkontroluje graf, zda je orientovany ci neorientovany 
// a ve spravnem formatu
//
//  GRAF_ORIENTOVANY    graf je orientovany
//  GRAF_NEORIENTOVANY  graf je neorientovany
//  GRAF_CHYBA          chyba formatu
//                      pro vsechny i musi platit, ze w(i,i) = graf[i][i] = 0 
unsigned kontrolaGrafu( unsigned ** graf, unsigned pocetUzlu ); // graf by mel byt const..

// vypise graf (formatovanou w-matici) do vystupniho streamu os
void vypisGrafu( ostream & os, unsigned ** graf, unsigned pocetUzlu ); // graf by mel byt const..


#endif // FUNKCE_SPOLECNE_aohiuefijn39nvkjns92

