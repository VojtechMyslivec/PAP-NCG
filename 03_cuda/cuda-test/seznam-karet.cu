#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int ct,dev;
    cudaError_t code;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&ct);
    code = cudaGetLastError();
    if(code)  printf("%s\n", cudaGetErrorString(code));


    if(ct == 0) {
        printf("Cuda device not found.\n");
        exit(0);
    }
    printf("Found %i Cuda device(s).\n",ct);

    for (dev = 0; dev < ct; ++dev) {
        printf("Cuda device %i\n", dev);

        cudaGetDeviceProperties(&prop,dev);
        printf("\tname : %s\n", prop.name);
        printf("\ttotalGlobablMem: %lu\n", (unsigned long)prop.totalGlobalMem);
        printf("\tsharedMemPerBlock: %i\n", (int)prop.sharedMemPerBlock);
        printf("\tregsPerBlock: %i\n", prop.regsPerBlock);
        printf("\twarpSize: %i\n", prop.warpSize);
        printf("\tmemPitch: %i\n", (int)prop.memPitch);
        printf("\tmaxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
        printf("\tmaxThreadsDim: %i, %i, %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\tmaxGridSize: %i, %i, %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\tclockRate: %i\n", prop.clockRate);
        printf("\ttotalConstMem: %i\n", (int)prop.totalConstMem);
        printf("\tmajor: %i\n", prop.major);
        printf("\tminor: %i\n", prop.minor);
        printf("\ttextureAlignment: %i\n", (int)prop.textureAlignment);
        printf("\tdeviceOverlap: %i\n", prop.deviceOverlap);
        printf("\tmultiProcessorCount: %i\n", prop.multiProcessorCount);
    }
}
