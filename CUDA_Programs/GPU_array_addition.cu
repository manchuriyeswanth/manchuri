#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;

__global__ void add(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}
void add(int *a,int *b,int *c)
{
    for (int i=0;i<5;i++)
    {
        c[i]=a[i]+b[i];
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec +ts.tv_nsec*1e-9;
}

int main()
{
    int a[5] = {1,2,3,4,5};
    int b[5] = {1,2,3,4,5};
    int *result;
    int c[5];
    result = (int*)malloc(5 * sizeof(int));
    struct timespec ts;

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, 5 * sizeof(int));
    cudaMalloc((void**)&d_b, 5 * sizeof(int));
    cudaMalloc((void**)&d_c, 5 * sizeof(int));

    cudaMemcpy(d_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

    double time1,time2;
    time1 = get_time();
    add(a,b,c);
    time2 = get_time();
    cout<<"CPU Time msec="<<(time2-time1)*1000<<endl;
    time1 = get_time();
    add<<<1, 5>>>(d_a, d_b, d_c, 5);
    time2 = get_time();
    cout<<"GPU Time msec="<<(time2-time1)*1000<<endl;

    cudaMemcpy(result, d_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
        cout << result[i] << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(result);

    return 0;
}

