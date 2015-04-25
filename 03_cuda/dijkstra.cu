/** dijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 algoritmus Dijkstra
 *
 *
 */

#include "dijkstra.h"
#include "cDijkstra.h"
#include "funkceSpolecne.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <omp.h>

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
    // TODO smazat? 
    // devDijkstra = new cDijkstra * [pocetUzlu];
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

void grafInicializaceNaGPU( unsigned ** graf, unsigned pocetUzlu, unsigned **& devGraf ) {
    // alokace matice -- pole sloupcu ---------------------
    HANDLE_ERROR( 
            cudaMalloc( 
                &devGraf,
                pocetUzlu*sizeof(*devGraf)
                )
            ); 

    // v cyklu se alokuji a kopiruji data z grafu na GPU
    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // alokace jednoho radku matice -------------------
        HANDLE_ERROR( 
                cudaMalloc( 
                    &devHodnoty,
                    pocetUzlu*sizeof(*devHodnoty)
                    )
                );
        // kopirovani jednoho radku matice ----------------
        HANDLE_ERROR( 
                cudaMemcpy( 
                    devHodnoty,
                    graf[i],
                    pocetUzlu*sizeof(*devHodnoty), 
                    cudaMemcpyHostToDevice 
                    )
                );
        // zkopirovani pointeru na radek do pole sloupcu --
        HANDLE_ERROR( 
                cudaMemcpy( 
                    &(devGraf[i]),
                    &(devHodnoty),
                    sizeof(devHodnoty),
                    cudaMemcpyHostToDevice
                    )
                );
    }
}

void inicializaceNtoN( unsigned ** graf,     unsigned pocetUzlu, 
                       unsigned **& vzdalenostM,
                       unsigned **& devGraf, cDijkstra **& devDijkstra
                     ) {
    // inicializace matic vysledku -------------------------
    vzdalenostM = new unsigned*[pocetUzlu];
    for ( unsigned i = 0; i < pocetUzlu; i++ ) {
        vzdalenostM[i] = new unsigned[pocetUzlu];
        for ( unsigned j = 0; j < pocetUzlu; j++ ) {
            vzdalenostM[i][j] = DIJKSTRA_NEKONECNO;
        }
    }

    // pseudo-staticka inicializace -----------------------
    grafInicializaceNaGPU( graf, pocetUzlu, devGraf );

    // inicializace objektu na GPU ------------------------
    dijkstraInicializaceNaGPU( devGraf, pocetUzlu, devDijkstra );
}

void uklidUkazatelu( unsigned **& dveDimenze, unsigned rozmer ) {
    if ( dveDimenze != NULL ) {
        for ( unsigned i = 0; i < rozmer; i++ ) {
            if ( dveDimenze[i] != NULL ) {
                delete [] dveDimenze[i];
                dveDimenze[i] = NULL;
            }
        }
        delete [] dveDimenze;
        dveDimenze = NULL;
    }
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
    // uvolneni objektu z pameti GPU ---------------------------------
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
        dijkstraObjektUklid( devDijkstra[idUzlu] );
    }
    // uvolneni pole ukazatelu na objekty na GPU --------------------
    HANDLE_ERROR( 
            cudaFree( devDijkstra )
            );
    devDijkstra = NULL;
}

void grafUklidNaGPU( unsigned **& devGraf, unsigned pocetUzlu ) {
    // v cyklu se kopiruji hodnoty ukazatelu na radek a ty se smazou
    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopirovani ukazatele z [pole sloupcu na device] --
        HANDLE_ERROR( 
                cudaMemcpy( 
                    &(devHodnoty),
                    &(devGraf[i]),
                    sizeof(devHodnoty),
                    cudaMemcpyDeviceToHost
                    )
                );
        // uvolneni pameti radku matice -------------------
        HANDLE_ERROR( 
                cudaFree( devHodnoty )
                );
    }
    // uvolneni pameti ukazatele na radky matice ----------
    HANDLE_ERROR( 
       cudaFree( devGraf )
    ); 
    devGraf = NULL;
}

void uklidNtoN( unsigned  ** vzdalenostM, 
                unsigned  ** devGraf,
                cDijkstra ** devDijkstra,
                unsigned     pocetUzlu 
              ) {
    uklidUkazatelu(     vzdalenostM, pocetUzlu );
    dijkstraUklidNaGPU( devDijkstra, pocetUzlu );
    grafUklidNaGPU(         devGraf, pocetUzlu );
}

void zkopirujDataZGPU( unsigned ** vzdalenostM, cDijkstra ** devDijkstra, unsigned pocetUzlu ) {
//     // zkopirovani pole [ukazatelu do device] ---------------------------
//     cDijkstra ** hostDevDijkstra = new cDijkstra* [pocetUzlu];
//     HANDLE_ERROR( 
//        cudaMemcpy( 
//                    hostDevDijkstra,
//                    devDijkstra,
//                    pocetUzlu*sizeof(*devDijkstra), 
//                    cudaMemcpyDeviceToHost 
//                  )
//     );

    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopiruje hodnotu ukazatele z [tridy na device] ----------
        HANDLE_ERROR( 
            cudaMemcpy( 
                        &(devHodnoty), 
                        //&(hostDevDijkstra[i]->vzdalenosti), 
                        &(devDijkstra[i]->vzdalenost), 
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
    
//     delete [] hostDevDijkstra;
}

__global__ void wrapperProGPU( cDijkstra ** devDijkstra, unsigned pocetUzlu ) {
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        devDijkstra[i]->devInicializujHodnoty();
        devDijkstra[i]->devSpustVypocet();
    }
    return;
}

bool dijkstraNtoN( unsigned ** graf, unsigned pocetUzlu, unsigned pocetVlaken ) {
    unsigned  ** devGraf;
    unsigned  ** vzdalenostM; 
    cDijkstra ** devDijkstra;

    inicializaceNtoN( graf, pocetUzlu, vzdalenostM, devGraf, devDijkstra );

    wrapperProGPU<<<1,1>>>( devDijkstra, pocetUzlu ) ;
    cudaDeviceSynchronize();

    zkopirujDataZGPU( vzdalenostM, devDijkstra, pocetUzlu );

    vypisVysledekMaticove( vzdalenostM, pocetUzlu );

    uklidNtoN( vzdalenostM, devGraf, devDijkstra, pocetUzlu );

    return true;
}

void vypisVysledekMaticove( unsigned ** vzdalenosti, unsigned pocetUzlu ) {
    cout << "Vzdalenosti:" << endl;
    vypisGrafu( cout, vzdalenosti, pocetUzlu );
}

