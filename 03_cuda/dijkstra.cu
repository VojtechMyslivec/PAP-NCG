/** dijkstra.cu
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-duben 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 algoritmus Dijkstra
 *
 *
 */

#include "dijkstra.cuh"
#include "cDijkstra.cuh"
#include "funkceSpolecne.cuh"

// TODO smazat
//#include <iostream>
//#include <fstream>
//#include <iomanip>
//#include <cstring>


using namespace std;

void dijkstraObjektInit( unsigned ** devGraf, unsigned pocetUzlu, unsigned idUzlu, cDijkstra *& devDijkstra ) {
    // vytvori objekt na host, aby ho zkopiroval na device ----------
    cDijkstra * hostDijkstra = new cDijkstra( pocetUzlu, idUzlu ); 

    cudaMalloc( &devDijkstra, sizeof(*devDijkstra) );
    cudaMemcpy( devDijkstra, hostDijkstra, sizeof(*devDijkstra), cudaMemcpyHostToDevice );

    delete hostDijkstra;

    // alokace vnitrnich poli ---------------------------------------
    unsigned * devVzdalenost;
    bool     * devUzavreny;
    HANDLE_ERROR( 
       cudaMalloc( 
                   &devVzdalenost,
                   pocetUzlu*sizeof(*devVzdalenost)
                 )
    );
    HANDLE_ERROR( 
       cudaMalloc( 
                   &devUzavreny,
                   pocetUzlu*sizeof(*devUzavreny)
                 )
    );

    // zkopirovani pointeru do instance -----------------------------
    HANDLE_ERROR( 
       cudaMemcpy( 
                   &(devDijkstra->vzdalenost),
                   &(devVzdalenost),
                   sizeof(devVzdalenost),
                   cudaMemcpyHostToDevice
                 )
    );
    HANDLE_ERROR( 
       cudaMemcpy( 
                   &(devDijkstra->uzavreny),
                   &(devUzavreny),
                   sizeof(devUzavreny),
                   cudaMemcpyHostToDevice
                 )
    );

    // zkopruje ukazatel na pseudo-staticky graf --------------------
    HANDLE_ERROR( 
       cudaMemcpy( 
                   &(devDijkstra->graf),
                   &(devGraf),
                   sizeof(devGraf),
                   cudaMemcpyHostToDevice
                 )
    );
}

void dijkstraInicializaceNaGPU( unsigned ** devGraf, unsigned pocetUzlu, cDijkstra **& devDijkstra ) {
    cDijkstra ** hostDevDijkstra = new cDijkstra * [pocetUzlu];

    // alokace objektu na GPU ---------------------------------------
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
        dijkstraObjektInit( devGraf, pocetUzlu, idUzlu, hostDevDijkstra[idUzlu] );
    }
    // alokace a zkopirovani pole ukazatelu na GPU ------------------
    HANDLE_ERROR( 
       cudaMalloc( 
                   &devDijkstra,
                   pocetUzlu*sizeof(*devDijkstra)
                 )
    );
    HANDLE_ERROR( 
       cudaMemcpy( 
                   devDijkstra,
                   hostDevDijkstra,
                   pocetUzlu*sizeof(*devDijkstra), 
                   cudaMemcpyHostToDevice 
                 )
    );

    delete [] hostDevDijkstra;
}

void inicializaceNtoN( unsigned **  graf, unsigned pocetUzlu, 
                       unsigned **& vzdalenostM,
                       unsigned **& devGraf, cDijkstra **& devDijkstra
                     ) {
    // inicializace matic vysledku -------------------------
    maticeInicializaceNaCPU( vzdalenostM, pocetUzlu );

    // zkopirovani grafu na GPU ----------------------------
    maticeInicializaceNaGPU( graf, pocetUzlu, devGraf );

    // inicializace objektu na GPU -------------------------
    dijkstraInicializaceNaGPU( devGraf, pocetUzlu, devDijkstra );

    // dalsi nastaveni pro GPU -----------------------------
#ifdef CACHE
    cudaFuncSetCacheConfig( wrapperProGPU, cudaFuncCachePreferL1 );
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif // CACHE
}

void dijkstraObjektUklid( cDijkstra *& devDijkstra ) {
    unsigned * devVzdalenost;
    bool     * devUzavreny;
    // zkopirovani pointeru z [instance na GPU] ---------------------
    HANDLE_ERROR( 
            cudaMemcpy( 
                &(devVzdalenost),
                &(devDijkstra->vzdalenost),
                sizeof(devVzdalenost),
                cudaMemcpyDeviceToHost
                )
            );
    HANDLE_ERROR( 
            cudaMemcpy( 
                &(devUzavreny),
                &(devDijkstra->uzavreny),
                sizeof(devUzavreny),
                cudaMemcpyDeviceToHost
                )
            );
    // uvolneni pameti vnitrnich poli instance ----------------------
    HANDLE_ERROR( 
            cudaFree( devVzdalenost )
            );
    HANDLE_ERROR( 
            cudaFree( devUzavreny )
            );

    // uvolni objekt z pameti na GPU --------------------------------
    cudaFree( devDijkstra );
    devDijkstra = NULL;
}

void dijkstraUklidNaGPU( cDijkstra **& devDijkstra, unsigned pocetUzlu ) {
    // zkopirovani pole [ukazatelu do device] ---------------------------
    cDijkstra ** hostDevDijkstra = new cDijkstra * [pocetUzlu];
    HANDLE_ERROR( 
       cudaMemcpy( 
                   hostDevDijkstra,
                   devDijkstra,
                   pocetUzlu*sizeof(*devDijkstra), 
                   cudaMemcpyDeviceToHost 
                 )
    );

    // uvolneni objektu z pameti GPU ---------------------------------
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
        dijkstraObjektUklid( hostDevDijkstra[idUzlu] );
    }
    // uvolneni pole ukazatelu na objekty na GPU --------------------
    HANDLE_ERROR( 
            cudaFree( devDijkstra )
            );
    devDijkstra = NULL;
    delete [] hostDevDijkstra;
}

void uklidNtoN( unsigned  ** vzdalenostM, 
                unsigned  ** devGraf,
                cDijkstra ** devDijkstra,
                unsigned     pocetUzlu 
              ) {
    maticeUklidNaCPU(   vzdalenostM, pocetUzlu );
    dijkstraUklidNaGPU( devDijkstra, pocetUzlu );
    maticeUklidNaGPU(       devGraf, pocetUzlu );
}

void zkopirujDataZGPU( unsigned ** vzdalenostM, cDijkstra ** devDijkstra, unsigned pocetUzlu ) {
    // zkopirovani pole [ukazatelu do device] ---------------------------
    cDijkstra ** hostDevDijkstra = new cDijkstra * [pocetUzlu];
    HANDLE_ERROR( 
            cudaMemcpy( 
                hostDevDijkstra,
                devDijkstra,
                pocetUzlu*sizeof(*devDijkstra), 
                cudaMemcpyDeviceToHost 
                )
            );

    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopiruje hodnotu ukazatele z [tridy na device] ----------
        HANDLE_ERROR( 
                cudaMemcpy( 
                    &(devHodnoty), 
                    &(hostDevDijkstra[i]->vzdalenost), 
                    //&(devDijkstra[i]->vzdalenost), 
                    sizeof(devHodnoty), 
                    cudaMemcpyDeviceToHost
                    )
                );

        // zkopiruje data z device do matice vzdalenosti ------------
        HANDLE_ERROR( 
                cudaMemcpy(
                    vzdalenostM[i],
                    devHodnoty,
                    pocetUzlu*sizeof(*devHodnoty),
                    cudaMemcpyDeviceToHost 
                    )
                );
    }

    delete [] hostDevDijkstra;
}

__global__ void wrapperProGPU( cDijkstra ** devDijkstra, unsigned pocetUzlu, unsigned pocetVlakenVBloku ) {
    unsigned blok   =  blockIdx.x;
    unsigned vlakno = threadIdx.x;
    unsigned i      = pocetVlakenVBloku * blok + vlakno;

#ifdef DEBUG
    printf( "thread id = %d, b = %d, v = %d\n", i, blok, vlakno );
#endif // DEBUG

    if ( i < pocetUzlu ) {
        devDijkstra[i]->devInicializujHodnoty();
        devDijkstra[i]->devSpustVypocet();
    }
    return;
}

void dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetWarpu ) {
    unsigned  ** devGraf;
    unsigned  ** vzdalenostM; 
    cDijkstra ** devDijkstra;
    // pocet vlaken v bloku -- minimalne pocet uzlu
    unsigned     vlaken = MIN( pocetWarpu * CUDA_WARP_VELIKOST, pocetUzlu );
    // horni cast pocetUzlu/vlaken
    unsigned     bloku  = ( pocetUzlu + vlaken - 1 ) / vlaken;

#ifdef MERENI
    // udalosti pro mereni casu vypoctu
    cudaEvent_t udalosti[MERENI_POCET];
    float       tVypocet, tCelkem;

    mereniInicializace( udalosti, MERENI_POCET);
    mereniZaznam( udalosti[MERENI_START] );
#endif // MERENI

    // inicializace a kopirovani dat na GPU --------------------------
    inicializaceNtoN( graf, pocetUzlu, vzdalenostM, devGraf, devDijkstra );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_ZAPIS] );
#endif // MERENI

    // vypocet na GPU ------------------------------------------------
    wrapperProGPU <<< bloku, vlaken >>> ( devDijkstra, pocetUzlu, vlaken ) ;
    HANDLE_ERROR(   cudaDeviceSynchronize( )        );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_VYPOCET] );
#endif // MERENI

    // kopirovani dat z GPU ------------------------------------------
    zkopirujDataZGPU( vzdalenostM, devDijkstra, pocetUzlu );

#ifdef MERENI
    mereniZaznam( udalosti[MERENI_KONEC] );
#endif // MERENI

#ifdef VYPIS
    // vypis vysledku ------------------------------------------------
    vypisGrafu( cout, vzdalenostM, pocetUzlu );
#endif // VYPIS

    // uvolneni pameti na CPU i GPU ----------------------------------
    uklidNtoN( vzdalenostM, devGraf, devDijkstra, pocetUzlu );

#ifdef MERENI
    mereniUplynulo( tVypocet, udalosti[MERENI_ZAPIS], udalosti[MERENI_VYPOCET] );
    mereniUplynulo(  tCelkem, udalosti[MERENI_START],   udalosti[MERENI_KONEC] );

    cerr << pocetUzlu << '	' << bloku   << '	' << vlaken << '	'
         << tVypocet  << '	' << tCelkem << endl;

    mereniUklid( udalosti, MERENI_POCET);

#endif // MERENI

}

