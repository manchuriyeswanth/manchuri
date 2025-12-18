#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;

struct gpu_mutex
{
    int *lock;
};
__host__ void mutex_init(gpu_mutex*);
__host__ void mutex_init(gpu_mutex *m)
{
    cudaMalloc((void**)&m->lock,sizeof(int));
    int *value =0;
    cudaMemcpy(&m->lock,value,sizeof(int),cudaMemcpyDeviceToHost);
}
__device__ void lock(gpu_mutex *m)
{   
    int i=0;
    while(atomicCAS(m->lock,0,1)!=0)
    {
        i++;
        if (i%10==0)
            printf("%d",i);
        //Spin Wait.
    }
}
__device__ void unlock (gpu_mutex *m)
{
    atomicExch(m->lock,0);
}
__global__ void mutex_kernel(gpu_mutex *m, int *c)
{   
    lock(m);
    int value = *c;
    *c = value + 1;
    printf("%d\n",*c);
    unlock(m);
}
int main()
{
    gpu_mutex m;
    mutex_init(&m);

    int *counter=0;
    cudaMalloc((void**)&counter,sizeof(int));
    cudaMemcpy(&counter,0,sizeof(int),cudaMemcpyHostToDevice);

    mutex_kernel<<<1,10>>>(&m,counter);

    int result;
    cudaMemcpy(&result,counter,sizeof(int),cudaMemcpyDeviceToHost);
    printf("%d\n",result);
    cudaFree(m.lock);
    cudaFree(counter);
    return 0;
}
