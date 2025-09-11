#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello from GPU\n");
}

int main(void)
{
    printf("Hello from CPU\n");

    // other code later
}
