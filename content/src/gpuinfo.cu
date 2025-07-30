#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int driverVersion;
    int deviceCount;
    int runtimeVersion;
    cudaError_t error_id;


    error_id = cudaDriverGetVersion(&driverVersion);
    if (error_id == cudaSuccess) {
        printf("CUDA Driver Version: %d.%d\n", driverVersion/1000, 
               (driverVersion % 100) / 10);
    } else {
        fprintf(stderr, "Error getting CUDA driver version\n");
    }

    error_id = cudaRuntimeGetVersion(&runtimeVersion);
    if (error_id == cudaSuccess) {
       printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, 
              (runtimeVersion % 100) / 10);
    } else {
       fprintf(stderr, "Error getting CUDA Runtime Version: %s\n", 
               cudaGetErrorString(error_id));
    }

    error_id  = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned %d\n", (int)error_id);
        fprintf(stderr, "Result: %s\n", cudaGetErrorString(error_id));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
    } else {
        printf("Found %d CUDA-capable device(s):\n", deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            printf("  Device %d: %s\n", i, deviceProp.name);
            printf("    Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
            printf("    Total Global Memory: %.2f GB\n", (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
            printf("    Multiprocessors: %d\n", deviceProp.multiProcessorCount);
            printf("    Clock Rate: %.2f MHz\n", (double)deviceProp.clockRate / 1000.0);
        }
    }

    return 0;
}

