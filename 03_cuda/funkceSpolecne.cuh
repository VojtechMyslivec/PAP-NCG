/** funkceSpolecne.cuh
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-kveten 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 Spolecne funkce pro nacitani / vypis dat
 *
 *
 */

#ifndef FUNKCE_SPOLECNE_aohiuefijn39nvkjns92
#define FUNKCE_SPOLECNE_aohiuefijn39nvkjns92


#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__) )
#define MIN( A, B )          ( A < B ? A : B )

#define CUDA_VYCHOZI_GPU_ID        0

#define CUDA_VYCHOZI_POCET_WARPU   4
#define CUDA_MAX_POCET_WARPU      32
#define CUDA_WARP_VELIKOST        32

// polovina z 32b unsigned -- tak, aby 2* nekonecno bylo stale dostatecne velke
// je to 2^30
#define NEKONECNO 0x40000000 

// velikost dlazdic, podle kterych se deli dlazdicovy algoritmus Floyd-Warshall
// optimalizovano pro 32, jelikoz pocet prvku v dlazdici je 32*32 = 1024
// tuto hodnotu nelze jen tak zmenit, bylo by potreba zmenit funkcne pro 
// prepocitavani pozice vlakna v matici
#define DLAZDICE_VELIKOST      32
#define DLAZDICE_VELIKOST_LOG2 5
#define DLAZDICE_VELIKOST_PUL            16 // DLAZDICE_VELIKOST / 2
#define DLAZDICE_VELIKOST_CTVRT           8 // DLAZDICE_VELIKOST / 4
#define DLAZDICE_VELIKOST_OSMINA          4 // DLAZDICE_VELIKOST / 8
#define DLAZDICE_VELIKOST_SESTNACTINA     2 // DLAZDICE_VELIKOST / 16

#define MAIN_OK            0
#define MAIN_ERR_USAGE     1
#define MAIN_ERR_VSTUP     2
#define MAIN_ERR_GRAF      3
#define MAIN_ERR_VYPOCET   4
#define MAIN_ERR_NO_GPU    5
#define MAIN_ERR_GPU_POCET 6
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

#include <iostream>
#include <climits>

using namespace std;

// error handle pro cuda funkce
void HandleError( cudaError_t chyba, const char * soubor, int radek );
   
// vypise usage na vystupni stream os, vola se s argumentem argv[0] 
// jako jmenem programu
void vypisUsage( ostream & os, const char * jmenoProgramu );

// smaze alokovany graf (provadi kontrolu ukazatele na NULL) a nastavi
// ukazatel(e) na NULL
void uklid( unsigned ** graf, unsigned pocetUzlu );

// nastavi cuda gpu na gpuID
// kod chyby MAIN_... ulozi do vystupni promenne navrat
bool nastavGpuID( int gpuID, unsigned & navrat );

// zkonstroluje, zda vstupni stream is je uz prazdny (muze obsahovat prazdne znaky)
//
//   true   stream je prazdny
//   false  stream neni/nebyl prazdny
bool zkontrolujPrazdnyVstup( istream & is );

// Funkce zajisti nacteni a kontrolu parametru.
// Do vystupnich promenych uklada:
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
bool parsujArgumenty( int argc, char ** argv, char *& souborSGrafem, unsigned & pocetWarpu, int & gpuID, unsigned & navrat );

// nacte jednu unsigned hodnotu ze vstupu
// pokud misto unsigned cisla nalezne - (nasledovanou prazdnym znakem)
// ulozi do hodnoty NEKONECNO
//
//   NACTI_OK           v poradku se podarilo nacist jednu unsigned hodnotu
//   NACTI_NEKONECNO    na vstupu byl znak '-' oddeleny mezerami, do hodnoty byla
//                      prirazena hodnota NEKONECNO
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
bool nactiData( const char * jmenoSouboru, unsigned ** & graf, unsigned & pocetUzlu, unsigned & velikostMatice );

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

// inicializuje a nakopiruje data grafu na GPU
//   pokud graf je NULL tak neinicializuje, pouze alokuje
void maticeInicializaceNaGPU( unsigned ** graf, unsigned pocetUzlu, unsigned **& devGraf );

// alokuje PAGE-LOCKED pamet pro rychlejsi kopirovani a inicializuje hodnoty na NEKONECNO
void maticeInicializaceNaCPU( unsigned **& hostMatice, unsigned pocetUzlu );

// uvolni data grafu na GPU
void maticeUklidNaGPU( unsigned **& devGraf, unsigned pocetUzlu );

// uvolni PAGE-LOCKED pamet 
void maticeUklidNaCPU( unsigned **& hostMatice, unsigned pocetUzlu );


#ifdef MERENI

// funkce pro mereni pomoci CUDA udalosti

// pro kazdou polozku z pole udalosti vytvori udalost (cudaEventCreate)
void mereniInicializace( cudaEvent_t udalosti[], unsigned pocet );

// zaznamena udalost a pocka na synchronizaci (cudaEventRecord a cudaDeviceSynchronize)
void mereniZaznam( cudaEvent_t udalost );

// zjisti, kolik uplynulo sekund mezi dvema udalostmi (cudaEventElapsedTime/1000)
void mereniUplynulo( float & cas, cudaEvent_t zacatek, cudaEvent_t konec );

// uvolni vytvorene udalosti (v poli) (cudaEventDestroy)
void mereniUklid( cudaEvent_t udalosti[], unsigned pocet );

#endif // MERENI

#endif // FUNKCE_SPOLECNE_aohiuefijn39nvkjns92

