/** floydWarshall.h
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni cast
 *                 funkce pro algoritmus Floyd-Warshall
 *
 *
 */

#ifndef FLOYDWARSHALL_kljnef29kjdsnf02
#define FLOYDWARSHALL_kljnef29kjdsnf02

#ifdef DEBUG2
   #define DEBUG
#endif // DEBUG2

#include "funkceSpolecne.h"
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

// funkce zabalujici kompletni vypocet vcetne inicializace a uklidu...
// vysledek vypise na stdout
void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken );

// funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
//void inicializace( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni, unsigned pocetVlaken );
void inicializace( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned pocetVlaken );

// funkce, ktera zajisti uklizeni alokovanych promennych
//void uklid( unsigned pocetUzlu, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni );
void uklid( unsigned pocetUzlu, unsigned **& delkaPredchozi, unsigned **& delkaAktualni );

// realizuje samotny (paralelni) vypocet algoritmu Floyd-Warshalla O( n^3 / p ) 
//void spustVypocet( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni, unsigned **& predchudcePredchozi, unsigned **& predchudceAktualni );
void spustVypocet( unsigned pocetUzlu, unsigned ** graf, unsigned **& delkaPredchozi, unsigned **& delkaAktualni );

// pomocna funkce prohazujici dva ukazatele
void prohodUkazatele( unsigned **& ukazatel1, unsigned **& ukazatel2 );

// funkce pro vypis matice delek a predchudcu
//void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka, unsigned ** predchudce );
void vypisVysledekMaticove( unsigned pocetUzlu, unsigned ** delka );

#endif // FLOYDWARSHALL_kljnef29kjdsnf02

