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

// funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& hostDelka, unsigned **& devDelka ) {
    // alokovani pameti pro vysledek -----------------------
    maticeInicializaceNaCPU( hostDelka, pocetUzlu );

    // alokovani pameti na GPU -----------------------------
    maticeInicializaceNaGPU(   graf, pocetUzlu, devDelka );

    // dalsi nastaveni pro GPU -----------------------------
#ifdef CACHE
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif // CACHE
}

// funkce, ktera zajisti uklizeni alokovanych promennych
void uklid( unsigned pocetUzlu, unsigned **& hostDelka, unsigned **& devDelka ) {
    maticeUklidNaCPU(       hostDelka, pocetUzlu );
    maticeUklidNaGPU(  devDelka, pocetUzlu );

    hostDelka    = NULL;
    devDelka = NULL;
}

void zkopirujDataZGPU( unsigned ** hostDelka, unsigned ** devDelka, unsigned pocetUzlu ) {
    // zkopirovani pole [ukazatelu do device] ---------------------------
    unsigned ** hostDevVzdalenost = new unsigned * [pocetUzlu];
    HANDLE_ERROR( 
            cudaMemcpy( 
                hostDevVzdalenost,
                devDelka,
                pocetUzlu*sizeof(*devDelka), 
                cudaMemcpyDeviceToHost 
                )
            );

    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopiruje data z device do matice vzdalenosti ------------
        HANDLE_ERROR( 
                cudaMemcpy(
                    hostDelka[i],
                    hostDevVzdalenost[i],
                    pocetUzlu*sizeof(*hostDevVzdalenost[i]),
                    cudaMemcpyDeviceToHost 
                    )
                );
    }

    delete [] hostDevVzdalenost;
}

// vzdy se spousti 32 bloku x 32 vlaken -- DLAZDICE_VELIKOST x DLAZDICE_VELIKOST 
__global__ void kernelProNezavisleDlazdice( unsigned ** devDelka, unsigned pocetUzlu, unsigned dlazdice, unsigned krok ) {
    const unsigned radek   =  blockIdx.x + ( dlazdice << DLAZDICE_VELIKOST_LOG2 );
    const unsigned sloupec = threadIdx.x + ( dlazdice << DLAZDICE_VELIKOST_LOG2 );
#ifdef DEBUG
    const unsigned blok   =  blockIdx.x;
    const unsigned vlakno = threadIdx.x;
    const unsigned id     =  blockDim.x * blok + vlakno;
    printf( "  - Kernel 1: vlakno id = %d, b = %d, v = %d, M[ %d , %d ]\n", id, blok, vlakno, radek, sloupec );
#endif // DEBUG

    if ( radek < pocetUzlu && sloupec < pocetUzlu ) {
        devDelka[radek][sloupec] = MIN(  devDelka[radek][sloupec],  devDelka[radek][krok] + devDelka[krok][sloupec]  );
    }
}

// vzdy se spousti pocetDlazdic (bloku) * 32 (x) *  32 vlaken (y) 
__global__ void kernelProRadky( unsigned ** devDelka, unsigned pocetUzlu, unsigned dlazdiceRadek, unsigned krok ) {
    const unsigned dlazdiceSloupec = blockIdx.x;
    const unsigned radek           = threadIdx.x + (   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2 );
    const unsigned sloupec         = threadIdx.y + ( dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2 );
#ifdef DEBUG
    const unsigned blok   =  blockIdx.x;
    const unsigned vlakno = threadIdx.x;
    const unsigned id     = DLAZDICE_VELIKOST * blok + vlakno;
    printf( "  - Kernel 2: vlakno id = %d, b = %d, v = %d, M[ %d , %d ]\n", id, blok, vlakno, radek, sloupec );
#endif // DEBUG

    if ( radek < pocetUzlu && sloupec < pocetUzlu ) {
        devDelka[radek][sloupec] = MIN(  devDelka[radek][sloupec],  devDelka[radek][krok] + devDelka[krok][sloupec]  );
    }
}

// vzdy se spousti pocetDlazdic (bloku) * 32 (x) *  32 vlaken (y) 
__global__ void kernelProSloupce( unsigned ** devDelka, unsigned pocetUzlu, unsigned dlazdiceSloupec, unsigned krok ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned radek           = threadIdx.x + (   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2 );
    const unsigned sloupec         = threadIdx.y + ( dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2 );
#ifdef DEBUG
    const unsigned blok   =  blockIdx.x;
    const unsigned vlakno = threadIdx.x;
    const unsigned id     = DLAZDICE_VELIKOST * blok + vlakno;
    printf( "  - Kernel 2: vlakno id = %d, b = %d, v = %d, M[ %d , %d ]\n", id, blok, vlakno, radek, sloupec );
#endif // DEBUG

    if ( radek < pocetUzlu && sloupec < pocetUzlu ) {
        devDelka[radek][sloupec] = MIN(  devDelka[radek][sloupec],  devDelka[radek][krok] + devDelka[krok][sloupec]  );
    }
}

__global__ void kernelProDvouZavisleDlazdice( unsigned ** devDelka, unsigned pocetUzlu, unsigned krok ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    const unsigned radek   = threadIdx.x + dlazdiceRadek   * DLAZDICE_VELIKOST;
    const unsigned sloupec = threadIdx.y + dlazdiceSloupec * DLAZDICE_VELIKOST;
#ifdef DEBUG
    const unsigned blok   =  blockIdx.x;
    const unsigned vlakno = threadIdx.x;
    const unsigned id     = DLAZDICE_VELIKOST * blok + vlakno;
    printf( "  - Kernel 2: vlakno id = %d, b = %d, v = %d, M[ %d , %d ]\n", id, blok, vlakno, radek, sloupec );
#endif // DEBUG

    if ( radek < pocetUzlu && sloupec < pocetUzlu ) {
        devDelka[radek][sloupec] = MIN(  devDelka[radek][sloupec],  devDelka[radek][krok] + devDelka[krok][sloupec]  );
    }
}

// realizuje samotny (paralelni) vypocet algoritmu Floyd-Warshalla O( n^3 / p ) 
void spustVypocet( unsigned ** devDelka, unsigned pocetUzlu, unsigned pocetWarpu ) {
    const unsigned s = DLAZDICE_VELIKOST;
    // horni cast pocetUzlu / s
    const unsigned pocetDlazdic = ( pocetUzlu + s - 1 ) / s; 
    const dim3     prvky2D( s, s );
    const dim3     dlazdice2D( pocetDlazdic, pocetDlazdic );

    for ( unsigned b = 0 ; b < pocetDlazdic ; b++ ) {
#ifdef DEBUG
        printf( "b = %d\n", b );
#endif // DEBUG
        // nezavisle dlazdice -- na hl. diagonale dlazdickovane matice ---------
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
            kernelProNezavisleDlazdice <<< s, s >>> ( devDelka, pocetUzlu, b, k );
            HANDLE_ERROR(   cudaDeviceSynchronize( ) );
        }

        // jedno-zavisle dlazdice ----------------------------------------------
        // ve stejnem radku
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
            kernelProRadky <<< pocetDlazdic, prvky2D >>> ( devDelka, pocetUzlu, b, k );
            HANDLE_ERROR(   cudaDeviceSynchronize( ) );
        }
        // ve stejnem sloupci
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
            kernelProSloupce <<< pocetDlazdic, prvky2D >>> ( devDelka, pocetUzlu, b, k );
            HANDLE_ERROR(   cudaDeviceSynchronize( ) );
        }

        // dvou-zavisle dlazdice -- zbytek -------------------------------------
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            if ( k >= pocetUzlu ) break;    // pokud je uz mimo, konci
            kernelProDvouZavisleDlazdice <<< dlazdice2D, prvky2D >>> ( devDelka, pocetUzlu, k );
        }

#ifdef DEBUG
        printf( "\n" );
#endif // DEBUG
    }
}

void floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu ) {
    unsigned ** devDelka  = NULL;
    unsigned ** hostDelka = NULL;
    
#ifdef MERENI
    // udalosti pro mereni casu vypoctu
    cudaEvent_t udalosti[MERENI_POCET];
    float       tVypocet, tCelkem;

    mereniInicializace( udalosti, MERENI_POCET);
    mereniZaznam( udalosti[MERENI_START] );
#endif // MERENI

    // inicializace a kopirovani dat na GPU --------------------------
    inicializace( graf, pocetUzlu, hostDelka, devDelka );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_ZAPIS] );
#endif // MERENI

    // vypocet na GPU ------------------------------------------------
    spustVypocet( devDelka, pocetUzlu, pocetWarpu );
    HANDLE_ERROR(   cudaDeviceSynchronize( )        );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_VYPOCET] );
#endif // MERENI

    // kopirovani dat z GPU ------------------------------------------
    zkopirujDataZGPU( hostDelka, devDelka, pocetUzlu );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_KONEC] );
#endif // MERENI

#ifdef VYPIS
    // vypis vysledku ------------------------------------------------
    vypisGrafu( cout, hostDelka, pocetUzlu );
#endif // VYPIS

    // uvolneni pameti na CPU i GPU ----------------------------------
    uklid( pocetUzlu, hostDelka, devDelka );

#ifdef MERENI
    mereniUplynulo( tVypocet, udalosti[MERENI_ZAPIS], udalosti[MERENI_VYPOCET] );
    mereniUplynulo(  tCelkem, udalosti[MERENI_START],   udalosti[MERENI_KONEC] );

    cerr << pocetUzlu << '	' //<< bloku   << '	' << vlakenVBloku << '	'
         << tVypocet  << '	' << tCelkem << endl;

    mereniUklid( udalosti, MERENI_POCET);
#endif // MERENI

}

