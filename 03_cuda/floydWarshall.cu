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
#ifdef DEBUG
   #include <stdio.h>
#endif // DEBUG

void inicializace( unsigned ** graf, unsigned pocetUzlu,
        unsigned **& vzdalenostM,
        unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni
        ) {
    // alokovani pameti pro vysledek -----------------------
    maticeInicializaceNaCPU( vzdalenostM, pocetUzlu );

    // alokovani pameti na GPU -----------------------------
    // devDelkaPredchozi bude kopie grafu
    maticeInicializaceNaGPU(   graf, pocetUzlu, devDelkaPredchozi );
    // devDelkaAktualni bude jen alokovana (prepisuje se v alg.)
    maticeInicializaceNaGPU(   NULL, pocetUzlu, devDelkaAktualni  );

    // dalsi nastaveni pro GPU -----------------------------
#ifdef CACHE
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif // CACHE
}

void uklid( unsigned pocetUzlu, unsigned **& vzdalenostM, unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni ) {
    maticeUklidNaCPU(       vzdalenostM, pocetUzlu );
    maticeUklidNaGPU( devDelkaPredchozi, pocetUzlu );
    maticeUklidNaGPU(  devDelkaAktualni, pocetUzlu );

    vzdalenostM    = NULL;
    devDelkaPredchozi = devDelkaAktualni = NULL;
}

void zkopirujDataZGPU( unsigned ** vzdalenostM, unsigned ** devVzdalenostM, unsigned pocetUzlu ) {
    // zkopirovani pole [ukazatelu do device] ---------------------------
    unsigned ** hostDevVzdalenost = new unsigned * [pocetUzlu];
    HANDLE_ERROR( 
            cudaMemcpy( 
                hostDevVzdalenost,
                devVzdalenostM,
                pocetUzlu*sizeof(*devVzdalenostM), 
                cudaMemcpyDeviceToHost 
                )
            );

    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopiruje data z device do matice vzdalenosti ------------
        HANDLE_ERROR( 
                cudaMemcpy(
                    vzdalenostM[i],
                    hostDevVzdalenost[i],
                    pocetUzlu*sizeof(*hostDevVzdalenost[i]),
                    cudaMemcpyDeviceToHost 
                    )
                );
    }

    delete [] hostDevVzdalenost;
}

void prohodUkazatele( unsigned **& ukazatel1, unsigned **& ukazatel2 ) {
    unsigned ** pomocny;
    pomocny   = ukazatel1;
    ukazatel1 = ukazatel2;
    ukazatel2 = pomocny;
}

void spustVypocet( unsigned **& devDelkaPredchozi, unsigned **& devDelkaAktualni, unsigned pocetUzlu, unsigned pocetBloku, unsigned pocetVlaken ) {
    for ( unsigned k = 0; k < pocetUzlu; k++ ) {
#ifdef DEBUG
        printf( "k = %d\n", k );
#endif // DEBUG
        // TODO strange casting?!
        wrapperProGPU <<< pocetBloku, pocetVlaken >>> ( (const unsigned **)devDelkaPredchozi, devDelkaAktualni, pocetUzlu, pocetVlaken, k );
        HANDLE_ERROR(   cudaDeviceSynchronize( ) );
#ifdef DEBUG
        printf( "\n" );
#endif // DEBUG

        // prohozeni predchozi a aktualni
        prohodUkazatele( devDelkaPredchozi, devDelkaAktualni );
    }

    // prohozeni predchozi a aktualni, aby vysledky byly v aktualnim ( po skonceni cyklu jsou vysledky v predchozim )
    prohodUkazatele( devDelkaPredchozi, devDelkaAktualni );
}

__global__ void wrapperProGPU( const unsigned ** devDelkaPredchozi, unsigned ** devDelkaAktualni, 
                               unsigned pocetUzlu, unsigned pocetVlakenVBloku, 
                               unsigned krok
                             ) {
    unsigned blok   =  blockIdx.x;
    unsigned vlakno = threadIdx.x;
    unsigned id     = pocetVlakenVBloku * blok + vlakno;
    unsigned novaVzdalenost;

    unsigned radek   = id / pocetUzlu;
    unsigned sloupec = id % pocetUzlu;

#ifdef DEBUG
    printf( "thread id = %d, b = %d, v = %d, M[ %d , %d ]\n", id, blok, vlakno, radek, sloupec );
#endif // DEBUG
    if ( radek < pocetUzlu ) {

        // osetreni nekonecna
        if ( devDelkaPredchozi[radek][krok] == NEKONECNO || devDelkaPredchozi[krok][sloupec] == NEKONECNO )
            novaVzdalenost = NEKONECNO;
        else
            novaVzdalenost = devDelkaPredchozi[radek][krok] + devDelkaPredchozi[krok][sloupec];

        // pokud nalezne kratsi cestu, zapise ji
        if ( novaVzdalenost < devDelkaPredchozi[radek][sloupec] ) {
            devDelkaAktualni[radek][sloupec] = novaVzdalenost;
        }// jinak delka zustava
        else {
            devDelkaAktualni[radek][sloupec] = devDelkaPredchozi[radek][sloupec];
        }

    }
}

void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu ) {
    unsigned ** devDelkaPredchozi = NULL;
    unsigned ** devDelkaAktualni  = NULL;
    unsigned ** vzdalenostM       = NULL;
    unsigned    pocetVlakenMin    = pocetUzlu * pocetUzlu;
    // pocet vlaken v bloku -- minimalne pocet uzlu ^ 2
    unsigned vlakenVBloku = MIN( pocetWarpu * CUDA_WARP_VELIKOST, pocetVlakenMin );
    // horni cast pocetUzlu/vlakenVBloku
    unsigned bloku        = ( pocetVlakenMin + vlakenVBloku - 1 ) / vlakenVBloku;

    
#ifdef MERENI
    // udalosti pro mereni casu vypoctu
    cudaEvent_t udalosti[MERENI_POCET];
    float       tVypocet, tCelkem;

    mereniInicializace( udalosti, MERENI_POCET);
    mereniZaznam( udalosti[MERENI_START] );
#endif // MERENI

    // inicializace a kopirovani dat na GPU --------------------------
    inicializace( graf, pocetUzlu, vzdalenostM, devDelkaPredchozi, devDelkaAktualni );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_ZAPIS] );
#endif // MERENI

    // vypocet na GPU ------------------------------------------------
    spustVypocet( devDelkaPredchozi, devDelkaAktualni, pocetUzlu, bloku, vlakenVBloku );
    HANDLE_ERROR(   cudaDeviceSynchronize( )        );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_VYPOCET] );
#endif // MERENI

    // kopirovani dat z GPU ------------------------------------------
    zkopirujDataZGPU( vzdalenostM, devDelkaAktualni, pocetUzlu );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_KONEC] );
#endif // MERENI

#ifdef VYPIS
    // vypis vysledku ------------------------------------------------
    vypisGrafu( cout, vzdalenostM, pocetUzlu );
#endif // VYPIS

    // uvolneni pameti na CPU i GPU ----------------------------------
    uklid( pocetUzlu, vzdalenostM, devDelkaPredchozi, devDelkaAktualni );

#ifdef MERENI
    mereniUplynulo( tVypocet, udalosti[MERENI_ZAPIS], udalosti[MERENI_VYPOCET] );
    mereniUplynulo(  tCelkem, udalosti[MERENI_START],   udalosti[MERENI_KONEC] );

    cerr << pocetUzlu << '	' << bloku   << '	' << vlakenVBloku << '	'
         << tVypocet  << '	' << tCelkem << endl;

    mereniUklid( udalosti, MERENI_POCET);
#endif // MERENI

}

