/** floydWarshall.cu
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

#include "floydWarshall.cuh"

void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu ) {
    unsigned ** devDelkaPredchozi = NULL;
    unsigned ** devDelkaAktualni  = NULL;
    unsigned ** vzdalenostM       = NULL;
    // pocet vlaken v bloku -- minimalne pocet uzlu
    int vlaken = MIN( pocetWarpu * CUDA_WARP_VELIKOST, pocetUzlu );
    // horni cast pocetUzlu/vlaken
    int bloku  = ( pocetUzlu + vlaken - 1 ) / vlaken;

    // inicializace a kopirovani dat na GPU --------------------------
    inicializace( graf, pocetUzlu, vzdalenostM, devDelkaPredchozi, devDelkaAktualni );

    // vypocet na GPU ------------------------------------------------
    wrapperProGPU<<<bloku,vlaken>>>( devDelkaPredchozi, devDelkaAktualni, pocetUzlu, vlaken );
    HANDLE_ERROR(   cudaDeviceSynchronize( )        );

    // kopirovani dat z GPU ------------------------------------------
    // TODO

#ifdef VYPIS
    // vypis vysledku ------------------------------------------------
    vypisGrafu( cout, devDelkaAktualni, pocetUzlu );
#endif // VYPIS

    // uvolneni pameti na CPU i GPU ----------------------------------
    uklid( pocetUzlu, vzdalenostM, devDelkaPredchozi, devDelkaAktualni );
}

void inicializace( unsigned ** graf, unsigned pocetUzlu,
        unsigned **& vzdalenostM,
        unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni
        ) {
    // alokovani pameti pro vysledek ------------------------
    maticeInicializaceNaCPU( vzdalenostM, pocetUzlu );
    // alokovani pameti na GPU ------------------------------
    // devDelkaPredchozi bude kopie grafu
    maticeInicializaceNaGPU(   graf, pocetUzlu, devDelkaPredchozi );
    // devDelkaAktualni bude jen alokovana (prepisuje se v alg.)
    maticeInicializaceNaGPU(   NULL, pocetUzlu, devDelkaAktualni  );
}

void uklid( unsigned pocetUzlu, unsigned **& vzdalenostM, unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni ) {
    maticeUklidNaCPU(     vzdalenostM, pocetUzlu );
    maticeUklidNaGPU( devDelkaPredchozi, pocetUzlu );
    maticeUklidNaGPU(  devDelkaAktualni, pocetUzlu );

    vzdalenostM    = NULL;
    devDelkaPredchozi = devDelkaAktualni = NULL;
}

void prohodUkazatele( unsigned **& ukazatel1, unsigned **& ukazatel2 ) {
    unsigned ** pomocny;
    pomocny   = ukazatel1;
    ukazatel1 = ukazatel2;
    ukazatel2 = pomocny;
}

// TODO
__device__ void spustVypocet( unsigned ** graf, unsigned pocetUzlu, unsigned **& delkaPredchozi, unsigned **& delkaAktualni ) {
    unsigned novaVzdalenost;

        for ( unsigned i = 0; i < pocetUzlu; i++ ) {
            for ( unsigned j = 0; j < pocetUzlu; j++ ) {
                // osetreni nekonecna
                if ( delkaPredchozi[i][k] == NEKONECNO || delkaPredchozi[k][j] == NEKONECNO )
                    novaVzdalenost = NEKONECNO;
                else
                    novaVzdalenost = delkaPredchozi[i][k] + delkaPredchozi[k][j];

                // pokud nalezne kratsi cestu, zapise ji
                if ( novaVzdalenost < delkaPredchozi[i][j] ) {
                    delkaAktualni[i][j] = novaVzdalenost;
                }// jinak delka zustava
                else {
                    delkaAktualni[i][j] = delkaPredchozi[i][j];
                }
            }
        }
}

// TODO
__global__ void wrapperProGPU( unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni, unsigned pocetUzlu, unsigned pocetVlakenVBloku ) {
    int blok   =  blockIdx.x;
    int vlakno = threadIdx.x;
    int i      = pocetVlakenVBloku * blok + vlakno;

#ifdef DEBUG
    printf( "thread id = %d, b = %d, v = %d\n", i, blok, vlakno );
#endif // DEBUG

    // TODO takhle fakt ne
    for ( unsigned k = 0; k < pocetUzlu; k++ ) {
        if ( i < pocetUzlu ) {
            devDijkstra[i]->devInicializujHodnoty();
            devDijkstra[i]->devSpustVypocet();
        }
        return;

        // prohozeni predchozi a aktualni
        prohodUkazatele( delkaPredchozi, delkaAktualni );
    }

    // prohozeni predchozi a aktualni, aby vysledky byly v aktualnim ( po skonceni cyklu jsou vysledky v predchozim )
    prohodUkazatele( delkaPredchozi, delkaAktualni );
}

