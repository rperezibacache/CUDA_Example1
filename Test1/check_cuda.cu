#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("CUDA library test:\n");
    
    // Test if we can access CUDA functions
    int runtime_version = 0;
    cudaError_t error = cudaRuntimeGetVersion(&runtime_version);
    
    if (error == cudaSuccess) {
        printf("CUDA Runtime version: %d\n", runtime_version);
        
        int deviceCount;
        error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess) {
            printf("Number of CUDA devices: %d\n", deviceCount);
        } else {
            printf("cudaGetDeviceCount error: %s\n", cudaGetErrorString(error));
        }
    } else {
        printf("cudaRuntimeGetVersion error: %s\n", cudaGetErrorString(error));
    }
    
    return 0;
}