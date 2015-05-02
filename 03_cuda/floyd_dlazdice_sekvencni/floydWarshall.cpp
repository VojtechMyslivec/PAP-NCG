/** floydWarshall.cpp
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

#include "floydWarshall.hpp"

inline void relaxace( unsigned ** delka, unsigned pocetUzlu, unsigned i, unsigned j, unsigned k ) {
//    if ( i >= pocetUzlu || j >= pocetUzlu || k >= pocetUzlu )
//        return;
    delka[i][j] = MIN(  delka[i][j],  delka[i][k] + delka[k][j]  );
}

// realizuje samotny (paralelni) vypocet algoritmu Floyd-Warshalla O( n^3 / p ) 
void spustVypocet( unsigned ** delka, unsigned pocetUzlu ) {
    unsigned s = DLAZDICE_VELIKOST;
    // horni cast pocetUzlu / s
    unsigned pocetDlazdic = ( pocetUzlu + s - 1 ) / s; 
    for ( unsigned b = 0 ; b < pocetDlazdic ; b++ ) {
        // nezavisle dlazdive -- na hl. diagonale dlazdickovane matice ---------
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
            for ( unsigned i = b*s ; i < (b+1)*s ; i++ ) {
                if ( i >= pocetUzlu ) break;        // pokud je uz mimo, konci
                for ( unsigned j = b*s ; j < (b+1)*s ; j++ ) {
                    if ( j >= pocetUzlu ) break;    // pokud je uz mimo, konci
                    relaxace( delka, pocetUzlu, i, j, k );
                }
            }
        }

        // jedno-zavisle dlazdice ----------------------------------------------
        // ve stejnem radku
        for ( unsigned ib = 0 ; ib < pocetDlazdic ; ib++ ) {
            if ( ib == b ) continue;    // pokud uz danou dlazdici spocital, preskoci
            for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
                if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
                for ( unsigned i = b*s ; i < (b+1)*s ; i++ ) {
                    if ( i >= pocetUzlu ) break;        // pokud je uz mimo, konci
                    for ( unsigned j = ib*s ; j < (ib+1)*s ; j++ ) {
                        if ( j >= pocetUzlu ) break;    // pokud je uz mimo, konci
                        relaxace( delka, pocetUzlu, i, j, k );
                    }
                }
            }
        }
        // ve stejnem sloupci
        for ( unsigned jb = 0 ; jb < pocetDlazdic ; jb++ ) {
            if ( jb == b ) continue;    // pokud uz danou dlazdici spocital, preskoci
            for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
                if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
                for ( unsigned i = jb*s ; i < (jb+1)*s ; i++ ) {
                    if ( i >= pocetUzlu ) break;        // pokud je uz mimo, konci
                    for ( unsigned j = b*s ; j < (b+1)*s ; j++ ) {
                        if ( j >= pocetUzlu ) break;    // pokud je uz mimo, konci
                        relaxace( delka, pocetUzlu, i, j, k );
                    }
                }
            }
        }

        // dvou-zavisle dlazdice -- zbytek -------------------------------------
        for ( unsigned ib = 0 ; ib < pocetDlazdic ; ib++ ) {
            if ( ib == b ) continue;        // pokud uz danou dlazdici spocital, preskoci
            for ( unsigned jb = 0 ; jb < pocetDlazdic ; jb++ ) {
                if ( jb == b ) continue;    // pokud uz danou dlazdici spocital, preskoci
                for ( unsigned i = jb*s ; i < (jb+1)*s ; i++ ) {
                    if ( i >= pocetUzlu ) break;            // pokud je uz mimo, konci
                    for ( unsigned j = ib*s ; j < (ib+1)*s ; j++ ) {
                        if ( j >= pocetUzlu ) break;        // pokud je uz mimo, konci
                        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
                            if ( k >= pocetUzlu ) break;    // pokud je uz mimo, konci
                            relaxace( delka, pocetUzlu, i, j, k );
                        }
                    }
                }
            }
        }

    }
}


void floydWarshall( unsigned ** graf, unsigned pocetUzlu ) {
    spustVypocet( graf, pocetUzlu);
#ifdef VYPIS
    vypisGrafu( cout, graf, pocetUzlu );
#endif // VYPIS
}


