#include <stdio.h>
#include <iostream>

#define DELKA           50
#define BLOK_VELIKOST   64

using namespace std;

__global__ void helloWorldParallel( void ) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    printf("Hello world from GPU %d/%d\n", j, i);
}

__global__ void sectiVektory( const int * vektorA, const int * vektorB, int * vektorV ) {
    int b = blockIdx.x;
    int t = threadIdx.x;
    int i = BLOK_VELIKOST*b+t;
    
    printf( "thread id = %d, b = %d, v = %d\n", i, b, t );

    if ( i < DELKA ) {
        vektorV[i] = vektorA[i]*3 + vektorB[i]*2;
        vektorV[i] *= vektorV[i]; 
    }
}


void vypisVektor( int * vektor ) {
    for (int i=0 ; i<DELKA ; i++ ) {
        cout << vektor[i] << ' ';
    }
    cout << endl;
}


bool vektorInit( int *& vektor ) {
    vektor = new int[DELKA];
    for ( int i = 0 ; i < DELKA ; i++ ) {
        vektor[i] = 2*i;
    }
    return true;
}

bool cudaVektorInit( int *& vektor ) {
    cudaMalloc( (void**)&vektor, DELKA*sizeof(int) );
    return true;
}


int main( void ) {
    printf("CPU hello\n");


    // Alokace
    int * hostA, * hostB, * hostC;
    int * devA,  * devB,  * devC;

    vektorInit( hostA );
    vektorInit( hostB );
    vektorInit( hostC );

    cudaVektorInit( devA );
    cudaVektorInit( devB );
    cudaVektorInit( devC );

    cudaMemcpy( devA, hostA, sizeof(float)*DELKA, cudaMemcpyHostToDevice );
    cudaMemcpy( devB, hostB, sizeof(float)*DELKA, cudaMemcpyHostToDevice );

    //helloWorldParallel<<<1,DELKA>>>( );
    int bloku = (DELKA + BLOK_VELIKOST - 1)/BLOK_VELIKOST;
    sectiVektory<<<bloku,BLOK_VELIKOST>>>( devA,devB, devC );
    cudaDeviceSynchronize(); //cudaDeviceReset(); // Aby CPU pockal na GPU (vypisovala)

    cudaMemcpy( hostC, devC, sizeof(float)*DELKA, cudaMemcpyDeviceToHost );
    
    vypisVektor( hostC );

    delete [] hostA;
    delete [] hostB;
    delete [] hostC;

    cudaFree( devA );
    cudaFree( devB );
    cudaFree( devC );

    printf("CPU bye\n");
}


