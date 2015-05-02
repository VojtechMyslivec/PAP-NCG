/** floydWarshall.cuh
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-kveten 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 funkce pro algoritmus Floyd-Warshall
 *
 *
 */

#ifndef FLOYDWARSHALL_kljnef29kjdsnf02
#define FLOYDWARSHALL_kljnef29kjdsnf02


#ifdef DEBUG2
    #define DEBUG
#endif // DEBUG2

#ifdef MERENI
    #define MERENI_POCET   4
    #define MERENI_START   0
    #define MERENI_ZAPIS   1
    #define MERENI_VYPOCET 2
    #define MERENI_KONEC   3
#endif // MERENI

// velikost dlazdic, podle kterych se deli dlazdicovy algoritmus Floyd-Warshall
// optimalizovano pro 32, jelikoz pocet prvku v dlazdici je 32*32 = 1024
// tuto hodnotu nelze jen tak zmenit, bylo by potreba zmenit funkcne pro 
// prepocitavani pozice vlakna v matici
#define DLAZDICE_VELIKOST 32

#include "funkceSpolecne.cuh"

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vysledek vypise na stdout
void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu );

#endif // FLOYDWARSHALL_kljnef29kjdsnf02

