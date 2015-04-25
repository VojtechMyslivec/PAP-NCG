
#include <stdio.h>
#include <iostream>

#define DELKA           16
#define BLOK_VELIKOST   32

#define CUDA_ALL __host__ __device__
#define CUDA_DEV __device__
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__) )

using namespace std;


class cVektor {
    public:
        int * hodnoty;

        cVektor();
        ~cVektor();
        CUDA_DEV void sectiSVektorem( const cVektor & vektor );
        CUDA_DEV void setHodnoty( const int * pole );
        void vypisHodnoty( );
};

cVektor::cVektor( ) {
    hodnoty = new int[DELKA];
    for( int i=0 ; i<DELKA ; i++ ) {
        hodnoty[i] = 2 * i;
    }
}

cVektor::~cVektor( ) {
    delete [] hodnoty;
}

CUDA_DEV void cVektor::setHodnoty( const int * pole ) {
    int blok  = blockIdx.x;
    int vlakno = threadIdx.x;
    int i = BLOK_VELIKOST * blok + vlakno;

    printf( "thread id = %d, b = %d, v = %d\n", i, blok, vlakno );

    if ( i < DELKA ) {
        hodnoty[i] = pole[i];
    }
}

CUDA_DEV void cVektor::sectiSVektorem( const cVektor & vektor ) {
    int blok  = blockIdx.x;
    int vlakno = threadIdx.x;
    int i = BLOK_VELIKOST * blok + vlakno;

    printf( "thread id = %d, b = %d, v = %d\n", i, blok, vlakno );

    if ( i < DELKA ) {
        hodnoty[i] += vektor.hodnoty[i];
    }
}

static void HandleError( cudaError_t chyba, const char * soubor, int radek ) {
    if ( chyba != cudaSuccess ) {
        printf( "%s v %s na radku %d\n", cudaGetErrorString( chyba ), soubor, radek );
        exit( 1 );
    }
}

void cVektor::vypisHodnoty( ) {
    for (int i=0 ; i<DELKA ; i++ ) {
        cout << hodnoty[i] << ' ';
    }
    cout << endl;
}



void cudaVektorInit( cVektor *& device, const cVektor * host  ) {
    cudaMalloc( (void**)&device, sizeof(*device) );
    if ( host != NULL ) {
        cudaMemcpy( device, host, sizeof(*device), cudaMemcpyHostToDevice );

        // hluboka kopie hodnot ----------------------
        // zkopiruje data na device
        int * devHodnoty;
        HANDLE_ERROR( 
           cudaMalloc( 
                       (void**)&devHodnoty,
                       DELKA*sizeof(*devHodnoty)
                     )
        );
        HANDLE_ERROR( 
           cudaMemcpy( 
                       devHodnoty,
                       host->hodnoty,
                       DELKA*sizeof(*devHodnoty), 
                       cudaMemcpyHostToDevice 
                     )
        );
        // zkopruje ukazatel na hodnoty na device
        HANDLE_ERROR( 
           cudaMemcpy( 
                       &(device->hodnoty),
                       &(devHodnoty),
                       sizeof(devHodnoty),
                       cudaMemcpyHostToDevice
                     )
        );
    }
}

bool cudaVektorUklid( cVektor *& device, cVektor * host  ) {
    bool ret = false;
    int * devHodnoty;
    // zkopiruje ukazatel z (tridy na device) do ukazale devHodnoty
    HANDLE_ERROR( 
        cudaMemcpy( 
                    &(devHodnoty), 
                    &(device->hodnoty), 
                    sizeof(devHodnoty), 
                    cudaMemcpyDeviceToHost
                  )
    );

    // pokud mam zkopirovat data
    if ( host != NULL ) {
        HANDLE_ERROR( 
           cudaMemcpy(
                       host->hodnoty,
                       devHodnoty,
                       DELKA*sizeof(*devHodnoty),
                       cudaMemcpyDeviceToHost 
                     )
        );
        ret = true;
    }

    HANDLE_ERROR( 
        cudaFree( devHodnoty )
    );
    HANDLE_ERROR( 
        cudaFree( device )
    );
    return ret;
}

__global__ void wrapperSectiVektory( cVektor * devA, const cVektor * devB, const int * devPole ) {
    printf( "CUDA HELLO\n" );
    devA->setHodnoty( devPole );
    devA->sectiSVektorem( *devB );
    printf( "CUDA bye\n" );
}

int main( void ) {
    printf("CPU hello\n");

    // Alokace
    cVektor * hostA = new cVektor();
    cVektor * hostB = new cVektor();
    
    int * hostPole = new int[DELKA];
    for ( int i = 0 ; i<DELKA ; i++ ) {
        hostPole[i] = i*i + i - 3;
    }
    int * devPole;
    HANDLE_ERROR( cudaMalloc( (void**)&devPole, sizeof(*devPole)*DELKA ) );
    HANDLE_ERROR( cudaMemcpy( devPole, hostPole, sizeof(*devPole)*DELKA, cudaMemcpyHostToDevice ) );

    cVektor * devA,  * devB;
    cudaVektorInit( devA, hostA );
    cudaVektorInit( devB, hostB );

    int bloku = (DELKA + BLOK_VELIKOST - 1)/BLOK_VELIKOST;

    wrapperSectiVektory<<<bloku,BLOK_VELIKOST>>>( devA, devB, devPole );
    HANDLE_ERROR( cudaDeviceSynchronize() ); 

    cudaVektorUklid( devA, hostA );
    cudaVektorUklid( devB, NULL  );
    HANDLE_ERROR( cudaFree( devPole ) );
    
    hostA->vypisHodnoty( );

    delete [] hostPole;
    delete hostA;
    delete hostB;

    printf("CPU bye\n");
}


