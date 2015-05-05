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

#include "funkceSpolecne.cuh"

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vysledek vypise na stdout
bool floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned velikostMatice, unsigned pocetWarpu );

#endif // FLOYDWARSHALL_kljnef29kjdsnf02

