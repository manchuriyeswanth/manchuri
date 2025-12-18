#include <iostream>
#include <cuda_runtime.h>
using namespace std;

void mmul(int* A, int* B, int* C, int M, int N, int K);
__global__ void cudamul(int *a, int *b, int *c, int M, int N, int K);
__global__ void cudaTiledMul(int *a, int *b, int *c, int M, int N, int K);
void checkerror(void);
#define BLOCKSIZE 256
#define TILE 64
int main()
{
    const int M = 20;
    const int N = 20;
    const int K = 15;

    int *A = new int[M*K];
    int *B = new int[K*N];
    int *C = new int[M*N];

    int *da,*db,*dc;
    cudaMalloc(&da, M*K*sizeof(int));
    cudaMalloc(&db, K*N*sizeof(int));
    cudaMalloc(&dc, M*N*sizeof(int));


    // Initialize A
    for (int i = 0; i < M*K; i++)
        A[i] = i;

    // Initialize B
    for (int i = 0; i < K*N; i++)
        B[i] = i;


    cudaMemcpy(da,A,M*K*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,B,K*N*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(da,A,M*K*sizeof(int),cudaMemcpyHostToDevice);
    // Matrix multiplication
    mmul(A, B, C, M, N, K);

    // Print result
    cout<<"CPU Result"<<endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            cout << C[i*N + j] << " ";
        cout << endl;
    }
    dim3 blocksize(BLOCKSIZE,BLOCKSIZE);
    dim3 numblocks(1,1);
    cudamul<<<numblocks,blocksize>>>(da,db,dc,M,N,K);
    checkerror();
    cudaMemcpy(C,dc,M*N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout<<"GPU Result"<<endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            cout << C[i*N + j] << " ";
        cout << endl;
    }


    dim3 TileBlockSize(TILE,TILE);
    dim3 numBlocksTile((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    

    cudaTiledMul<<<numBlocksTile,TileBlockSize>>>(da,db,dc,M,N,K);
    checkerror();
    cudaMemcpy(C,dc,M*N*sizeof(int),cudaMemcpyDeviceToHost);
    // Print result
    cudaDeviceSynchronize();

     cout<<"GPU Result Tiled Multiplication"<<endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            cout << C[i*N + j] << " ";
        cout << endl;
    }
    
    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}

void checkerror(void)
{
    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}
void mmul(int* A, int* B, int* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i*N + j] = 0;
            for (int k = 0; k < K; k++)
                C[i*N + j] += A[i*K + k] * B[k*N + j];
        }
    }
}

__global__ void cudamul(int *a, int *b, int *c, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {   
        int sum =0;
        for (int k =0 ;k < K ;k++)
        {
            sum += a[row*K+k]*b[k*N+col];
        }
        __syncthreads();
        c[row*N+col] = sum;
    }
}
__global__ void cudaTiledMul(int *a, int *b, int *c, int M, int N, int K)
{   
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int A[TILE][TILE];
    __shared__ int B[TILE][TILE];
    
    int sum =0;
    for( int t =0 ;t < (K+TILE-1)/TILE; t++)
    {
        if (ty < M && t*TILE+tx < K) 
        {
            A[ty][tx] = a[ty*K+t*TILE+tx];
        }
        else
        {
            A[ty][tx] = 0;
        }
        if (tx < N && t *TILE+ty < K)
        {
            B[ty][tx] = b[(t*TILE+ty)*N+tx];
        }
        else
        {

            B[ty][tx] =0;
        }
                __syncthreads();
        for (int k=0;k < TILE;k++)
        {
            sum += A[ty][k]*B[k][tx];
        }
                __syncthreads();
        if (ty < M && tx < N)
            c[ty*N+tx] = sum;
    }
    
}

