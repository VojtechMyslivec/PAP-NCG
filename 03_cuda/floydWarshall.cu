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
    // mam pocitat blok, ktery spocital pro nezavisly blok
    if ( dlazdiceRadek == dlazdiceSloupec ) {
        return;
    }
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
    // mam pocitat blok, ktery spocital pro nezavisly blok
    if ( dlazdiceRadek == dlazdiceSloupec ) {
        return;
    }
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

// pro tricet dva warpu -- 1024 vlaken <<< (N/s,N/s), (32,32)  >>>
__global__ void kernelProDvouZavisleDlazdice1024( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned sloupecSkupina  = threadIdx.y; // {0..31}
    const unsigned offsetSloupec   = sloupecSkupina;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka;

    rDelka = devDelka[radek][sloupec];

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // nakopirovani do sdilene pameti
        __syncthreads( ); // je treba synchronizace -- vola se 32 warpu
        if      ( sloupecSkupina == 0 )  // jedna skupina nakopiruje radek
            sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        else if ( sloupecSkupina == 1 )  // a druha skupina nakopiruje sloupec
            sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
                                         // ostatni jen cekaji -- nejde stejne lepe jak na 2 "takty"
        __syncthreads( ); // je treba synchronizace -- vola se 32 warpu
#endif // SHARED
#ifndef SHARED
        rDelka = MIN(  rDelka,  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec]  );
#endif // SHARED
#ifdef SHARED
        rDelka = MIN(  rDelka,  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][offsetSloupec]  );
#endif // SHARED
    }

    devDelka[radek][sloupec] = rDelka;
}

// pro sestnact warpu -- 512 vlaken <<< (N/s,N/s), (32,16)  >>>
__global__ void kernelProDvouZavisleDlazdice512( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned sloupecSkupina  = threadIdx.y; // {0..15}
    const unsigned offsetSloupec   = sloupecSkupina * DLAZDICE_VELIKOST_SESTNACTINA;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka[1][DLAZDICE_VELIKOST_SESTNACTINA];

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_SESTNACTINA ; j++ ) {
        rDelka[0][j] = devDelka[radek][sloupec + j];
    }

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // nakopirovani do sdilene pameti
        __syncthreads( ); // je treba synchronizace -- vola se 16 warpu
        if      ( sloupecSkupina == 0 )  // jedna skupina nakopiruje radek
            sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        else if ( sloupecSkupina == 1 )  // a druha skupina nakopiruje sloupec
            sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
                                         // ostatni jen cekaji -- nejde stejne lepe jak na 2 "takty"
        __syncthreads( ); // je treba synchronizace -- vola se 16 warpu
#endif // SHARED
        for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_SESTNACTINA; j++ ) {
#ifndef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec+j]  );
#endif // SHARED
#ifdef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][j+offsetSloupec]  );
#endif // SHARED
        }
    }

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_SESTNACTINA ; j++ ) {
        devDelka[radek][sloupec + j] = rDelka[0][j];
    }
}

// pro osm warpu -- 256 vlaken <<< (N/s,N/s), (32,8)  >>>
__global__ void kernelProDvouZavisleDlazdice256( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned sloupecSkupina  = threadIdx.y; // {0..7}
    const unsigned offsetSloupec   = sloupecSkupina * DLAZDICE_VELIKOST_OSMINA;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka[1][DLAZDICE_VELIKOST_OSMINA];

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_OSMINA ; j++ ) {
        rDelka[0][j] = devDelka[radek][sloupec + j];
    }

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // nakopirovani do sdilene pameti
        __syncthreads( ); // je treba synchronizace -- vola se 8 warpu
        if      ( sloupecSkupina == 0 )  // jedna skupina nakopiruje radek
            sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        else if ( sloupecSkupina == 1 )  // a druha skupina nakopiruje sloupec
            sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
                                         // ostatni jen cekaji -- nejde stejne lepe jak na 2 "takty"
        __syncthreads( ); // je treba synchronizace -- volaji se 8 warpu
#endif // SHARED
        for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_OSMINA; j++ ) {
#ifndef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec+j]  );
#endif // SHARED
#ifdef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][j+offsetSloupec]  );
#endif // SHARED
        }
    }

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_OSMINA ; j++ ) {
        devDelka[radek][sloupec + j] = rDelka[0][j];
    }
}

// pro ctyri warpy -- 128 vlaken <<< (N/s,N/s), (32,4)  >>>
__global__ void kernelProDvouZavisleDlazdice128( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned sloupecSkupina  = threadIdx.y; // {0..3}
    const unsigned offsetSloupec   = sloupecSkupina * DLAZDICE_VELIKOST_CTVRT;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka[1][DLAZDICE_VELIKOST_CTVRT];

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_CTVRT ; j++ ) {
        rDelka[0][j] = devDelka[radek][sloupec + j];
    }

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // nakopirovani do sdilene pameti
        __syncthreads( ); // je treba synchronizace -- volaji se 4 warpy
        if      ( sloupecSkupina == 0 )  // jedna skupina nakopiruje radek
            sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        else if ( sloupecSkupina == 1 )  // a druha skupina nakopiruje sloupec
            sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
                                         // ostatni jen cekaji -- nejde stejne lepe jak na 2 "takty"
        __syncthreads( ); // je treba synchronizace -- volaji se 4 warpy
#endif // SHARED
        for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_CTVRT; j++ ) {
#ifndef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec+j]  );
#endif // SHARED
#ifdef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][j+offsetSloupec]  );
#endif // SHARED
        }
    }

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_CTVRT ; j++ ) {
        devDelka[radek][sloupec + j] = rDelka[0][j];
    }
}

// pro dva warpy -- 64 vlaken <<< (N/s,N/s), (32,2)  >>>
__global__ void kernelProDvouZavisleDlazdice64( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned sloupecSkupina  = threadIdx.y; // {0,1}
    const unsigned offsetSloupec   = sloupecSkupina * DLAZDICE_VELIKOST_PUL;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka[1][DLAZDICE_VELIKOST_PUL];

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_PUL ; j++ ) {
        rDelka[0][j] = devDelka[radek][sloupec + j];
    }

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // nakopirovani do sdilene pameti
        __syncthreads( ); // je treba synchronizace -- volaji se 2 warpy
        if      ( sloupecSkupina == 0 )  // jedna skupina nakopiruje radek
            sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        else // ( sloupecSkupina == 1 )  // a druha skupina nakopiruje sloupec
            sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
        __syncthreads( ); // je treba synchronizace -- volaji se 2 warpy
#endif // SHARED
        for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_PUL ; j++ ) {
#ifndef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec+j]  );
#endif // SHARED
#ifdef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][j+offsetSloupec]  );
#endif // SHARED
        }
    }

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST_PUL ; j++ ) {
        devDelka[radek][sloupec + j] = rDelka[0][j];
    }
}

// pro jeden warp -- 32 vlaken -- <<< (N/s,N/s), (32,1) >>>
__global__ void kernelProDvouZavisleDlazdice32( unsigned ** devDelka, unsigned pocetUzlu, unsigned b ) {
    const unsigned dlazdiceRadek   = blockIdx.x;
    const unsigned dlazdiceSloupec = blockIdx.y;
    // mam pocitat blok, ktery spocitaly jine kernely
    if ( dlazdiceRadek == b || dlazdiceSloupec == b ) {
        return;
    }
    const unsigned offsetK         = b * blockDim.x;
    const unsigned dlazdiceRadekZaklad   =   dlazdiceRadek << DLAZDICE_VELIKOST_LOG2;
    const unsigned dlazdiceSloupecZaklad = dlazdiceSloupec << DLAZDICE_VELIKOST_LOG2;
    const unsigned offsetRadek     = threadIdx.x; // {0..31}
    const unsigned offsetSloupec   = 0;
    const unsigned radek           =   offsetRadek + dlazdiceRadekZaklad;
    const unsigned sloupec         = offsetSloupec + dlazdiceSloupecZaklad;
#ifdef SHARED
    __shared__ unsigned   sDelkaRadek[1][DLAZDICE_VELIKOST];
    __shared__ unsigned sDelkaSloupec[DLAZDICE_VELIKOST][1];
#endif // SHARED
               unsigned        rDelka[1][DLAZDICE_VELIKOST];

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST ; j++ ) {
        rDelka[0][j] = devDelka[radek][sloupec + j];
    }

    for ( unsigned k = 0 ; k < DLAZDICE_VELIKOST ; k++ ) { 
#ifdef SHARED
        // neni treba, kazde vlakno ma prave 1 prvek
        sDelkaRadek[0][offsetRadek]   = devDelka[k+offsetK][dlazdiceSloupecZaklad+offsetRadek]; // kazde vlakno se stara o jeden prvek
        sDelkaSloupec[offsetRadek][0] = devDelka[dlazdiceRadekZaklad+offsetRadek][k+offsetK];   // kazde vlakno se stara o jeden prvek
#endif // SHARED
        for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST ; j++ ) {
#ifndef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  devDelka[radek][k+offsetK] + devDelka[k+offsetK][sloupec+j]  );
#endif // SHARED
#ifdef SHARED
            rDelka[0][j] = MIN(  rDelka[0][j],  sDelkaSloupec[offsetRadek][0] + sDelkaRadek[0][j]  );
#endif // SHARED
        }
    }

    for ( unsigned j = 0 ; j < DLAZDICE_VELIKOST ; j++ ) {
        devDelka[radek][sloupec + j] = rDelka[0][j];
    }
}

// realizuje samotny (paralelni) vypocet algoritmu Floyd-Warshalla O( n^3 / p ) 
// pocetUzlu musi byt nasobkem DLAZDICE_VELIKOST jinak bude hazet SIGSEGV
void spustVypocet( unsigned ** devDelka, unsigned pocetUzlu, unsigned pocetWarpu, unsigned & bloku, unsigned & vlakenVBloku ) {
    const unsigned s = DLAZDICE_VELIKOST;
    // horni cast pocetUzlu / s
    const unsigned pocetDlazdic = ( pocetUzlu + s - 1 ) / s; 
    const dim3    prvky2D( s, s );
    const dim3 dlazdice2D( pocetDlazdic, pocetDlazdic );
    const dim3     blok2D( s, pocetWarpu );
    // jen vypocet pro mereni
    bloku                  = pocetDlazdic * pocetDlazdic;
    vlakenVBloku           = s * pocetWarpu;

    for ( unsigned b = 0 ; b < pocetDlazdic ; b++ ) {
#ifdef DEBUG
        printf( "b = %d\n", b );
#endif // DEBUG
        // nezavisle dlazdice -- na hl. diagonale dlazdickovane matice ---------
        for ( unsigned k = b*s ; k < (b+1)*s ; k++ ) {
            //if ( k >= pocetUzlu ) break;            // pokud je uz mimo, konci
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
        switch ( pocetWarpu ) {
            case 1:
                kernelProDvouZavisleDlazdice32   <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            case 2:
                kernelProDvouZavisleDlazdice64   <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            case 4:
                kernelProDvouZavisleDlazdice128  <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            case 8:
                kernelProDvouZavisleDlazdice256  <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            case 16:
                kernelProDvouZavisleDlazdice512  <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            case 32:
                kernelProDvouZavisleDlazdice1024 <<< dlazdice2D, blok2D  >>> ( devDelka, pocetUzlu, b );
                break;
            default:
                cerr << "spustVypocet(): nepodporovany pocet warpu!" << endl;
                return;
        }
        HANDLE_ERROR(   cudaDeviceSynchronize( ) );

#ifdef DEBUG
        printf( "\n" );
#endif // DEBUG
    }
}

// funkce pro inicializovani veskerych promennych potrebnych behem vypoctu 
void inicializace( unsigned ** graf, unsigned pocetUzlu, unsigned **& hostDelka, unsigned **& devDelka ) {
    // alokovani pameti pro vysledek -----------------------
    maticeInicializaceNaCPU( hostDelka, pocetUzlu );

    // alokovani pameti na GPU -----------------------------
    maticeInicializaceNaGPU(   graf, pocetUzlu, devDelka );

    // dalsi nastaveni pro GPU -----------------------------
#ifdef CACHE
    cudaFuncSetCacheConfig( kernelProNezavisleDlazdice,     cudaFuncCachePreferL1 );
    cudaFuncSetCacheConfig( kernelProRadky,                 cudaFuncCachePreferL1 );
    cudaFuncSetCacheConfig( kernelProSloupce,               cudaFuncCachePreferL1 );
#ifndef SHARED
    cudaFuncSetCacheConfig( kernelProDvouZavisleDlazdice32, cudaFuncCachePreferL1 );
    cudaFuncSetCacheConfig( kernelProDvouZavisleDlazdice64, cudaFuncCachePreferL1 );
#endif // SHARED
#ifdef SHARED
    cudaFuncSetCacheConfig( kernelProDvouZavisleDlazdice32, cudaFuncCachePreferShared );
    cudaFuncSetCacheConfig( kernelProDvouZavisleDlazdice64, cudaFuncCachePreferShared );
#endif // SHARED
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


bool floydWarshall( unsigned ** graf, unsigned pocetUzlu, unsigned velikostMatice, unsigned pocetWarpu ) {
    if ( pocetWarpu != 1 && pocetWarpu != 2 && pocetWarpu != 4 && pocetWarpu != 8 && pocetWarpu != 16 && pocetWarpu != 32 ) {
        cerr << "floydWarshall(): nepodporovany pocet warpu! ( w = " << pocetWarpu << " )" << endl;
        return false;
    }

    unsigned ** devDelka  = NULL;
    unsigned ** hostDelka = NULL;
    unsigned    bloku, vlakenVBloku;
    
#ifdef MERENI
    // udalosti pro mereni casu vypoctu
    cudaEvent_t udalosti[MERENI_POCET];
    float       tVypocet, tCelkem;

    mereniInicializace( udalosti, MERENI_POCET);
    mereniZaznam( udalosti[MERENI_START] );
#endif // MERENI

    // inicializace a kopirovani dat na GPU --------------------------
    inicializace( graf, velikostMatice, hostDelka, devDelka );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_ZAPIS] );
#endif // MERENI

    // vypocet na GPU ------------------------------------------------
    spustVypocet( devDelka, velikostMatice, pocetWarpu, bloku, vlakenVBloku );
    HANDLE_ERROR(   cudaDeviceSynchronize( )        );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_VYPOCET] );
#endif // MERENI

    // kopirovani dat z GPU ------------------------------------------
    // staci pocet uzlu, nezajima nas jak to dopadlo s vyplni na DLAZDICE_VELIKOST
    zkopirujDataZGPU( hostDelka, devDelka, pocetUzlu );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_KONEC] );
#endif // MERENI

#ifdef VYPIS
    // vypis vysledku ------------------------------------------------
    vypisGrafu( cout, hostDelka, pocetUzlu );
#endif // VYPIS

    // uvolneni pameti na CPU i GPU ----------------------------------
    uklid( velikostMatice, hostDelka, devDelka );

#ifdef MERENI
    mereniUplynulo( tVypocet, udalosti[MERENI_ZAPIS], udalosti[MERENI_VYPOCET] );
    mereniUplynulo(  tCelkem, udalosti[MERENI_START],   udalosti[MERENI_KONEC] );

    cerr << pocetUzlu << '	' << bloku   << '	' << vlakenVBloku << '	'
         << tVypocet  << '	' << tCelkem << endl;

    mereniUklid( udalosti, MERENI_POCET);
#endif // MERENI

    return true;
}

