/** floydWarshall.hpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <zdenek.novy@fit.cvut.cz>,       FIT CVUT v Praze
 *              
 * Datum:       unor-kveten 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 Algoritmus Floyd-Warshall pomoci dlazdickovani
 *                 funkce pro algoritmus Floyd-Warshall
 *
 *
 */

#ifndef FLOYDWARSHALL_kljnef29kjdsnf02
#define FLOYDWARSHALL_kljnef29kjdsnf02


#ifdef DEBUG2
    #define DEBUG
#endif // DEBUG2

#define DLAZDICE_VELIKOST 32

#include "funkceSpolecne.hpp"

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vysledek vypise na stdout
void floydWarshall( unsigned ** graf, unsigned pocetUzlu );

#endif // FLOYDWARSHALL_kljnef29kjdsnf02

