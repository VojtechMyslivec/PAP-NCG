

#include <stdio.h>
#include <iostream>

#define DELKA 65536

using namespace std;


void sectiVektory( const int * vektorA, const int * vektorB, int * vektorV ) {
    for( int i=0 ; i<DELKA ; i++ ) {
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


int main( void ) {
    printf("CPU hello\n");


    // Alokace
    int * hostA, * hostB, * hostC;

    vektorInit( hostA );
    vektorInit( hostB );
    vektorInit( hostC );

    sectiVektory( hostA,hostB, hostC );

    
    vypisVektor( hostC );

    delete [] hostA;
    delete [] hostB;
    delete [] hostC;

    printf("CPU bye\n");
}





