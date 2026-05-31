#include <iostream>
#include <cuda_runtime.h>
#define TILE 16

__global__ void tiled_matmul_qk(const float* Q, const float* K, float* S, int n, int d){
    __shared__ float sQ[TILE][TILE];
    __shared__ float sK[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Tile over embedding dimension d -- columns
    //Each thread loads an element of the tile, in each iteration
    for(int t=0;t<(d+TILE-1)/TILE;t++){
        int q_col = t*TILE+threadIdx.x;
        int k_col = t*TILE+threadIdx.y;

        // Load Q tile: Q[row][q_col]
        if(row < n && q_col < d) 
            sQ[threadIdx.y][threadIdx.x] = Q[row*d+q_col];
        else 
            sQ[threadIdx.y][threadIdx.x] = 0.0f;

        //Load K[col][k] -- local transpose instead of global transpose
        if(col < n && k_col < d) 
            sK[threadIdx.x][threadIdx.y] = K[col*d+k_col];
        else 
            sK[threadIdx.x][threadIdx.y] = 0.0f;
        __syncthreads();

        for(int k=0;k<TILE;k++){
            value += sQ[threadIdx.y][k] * sK[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row<n && col<n) 
        S[row*n+col] = value;
}

//one block per row?
//need to use shared memory for reductions
// max calculation -- Reduction ; Sum calc -- Reduction ; Normalization
//Shared memory size is defined during the kernel launch
__global__ void softmax(float* x, int n){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* row_start = x+(row*n);
    //how to set negative inf in CUDA?
    float row_max = -1e20f;

    //What happens when created w/o a definite size?
    extern __shared__ float shmem[];

    //1st pass -- calculate row max
    //all threads in the block sweep over the input row
    for(int j=tid;j<n;j+=blockDim.x){
        row_max = fmaxf(row_max,row_start[j]);
    }

    //sync threads to ensure that thread specific row_max is obtained
    shmem[tid]=row_max;
    __syncthreads();

    //Parallely reducing the maximum across all threads using shared memory
    for(int red=blockDim.x/2;red>0;red=red/2){
        if(tid<red){
            shmem[tid] = fmaxf(shmem[tid],shmem[tid+red]);
        }
        //need to ensure that all threads have reached this point before the next iteration
        __syncthreads();
    }
    row_max = shmem[0];

    //2nd pass -- calculating exponentials
    float exp_sum = 0.0;

    for(int i=tid;i<n;i+=blockDim.x){
        //the exp needs to be re-used during the next pass
        row_start[i] = expf(row_start[i]-row_max);
        exp_sum += row_start[i]; 
    }
    //thread specific exp sum
    shmem[tid]=exp_sum;
    //experienced race conditions
    __syncthreads();

    //Parallely reducing the exp sum
    for(int red=blockDim.x/2;red>0;red=red/2){
        if(tid<red){
            shmem[tid] += shmem[tid+red];
        }
        //need to ensure that all threads have reached this point before the next iteration
        __syncthreads();
    }
    exp_sum = shmem[0];

    //3rd pass -- normalization
    for(int i=tid;i<n;i+=blockDim.x){
        row_start[i] = row_start[i]/exp_sum;
    }
}

__global__ void tiled_matmul_pv(const float* P, const float* V, float* O, int n, int d){
    __shared__ float sP[TILE][TILE];
    __shared__ float sV[TILE][TILE];

    int row = blockIdx.y*TILE+threadIdx.y;
    int col = blockIdx.x*TILE+threadIdx.x;

    float value = 0.0;

    // Tile over shared inner dimension n
    for(int t=0; t<(n+TILE-1)/TILE;t++){
        int p_col = t*TILE+threadIdx.x;
        int v_row = t*TILE+threadIdx.y;

        if(row<n && p_col<n) 
            sP[threadIdx.y][threadIdx.x] = P[row*n+p_col];
        else 
            sP[threadIdx.y][threadIdx.x] = 0.0f;

        if(v_row < n && col < d) 
            sV[threadIdx.y][threadIdx.x] = V[v_row*d+col];
        else   
            sV[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for(int k=0;k<TILE;k++){
            value += sP[threadIdx.y][k] * sV[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row<n && col<d) 
        O[row*d+col] = value;
}


int main(){
    float *Q,*K,*V,*S,*O;
    float *dQ,*dK,*dV,*dS,*dO,gpu_time;
    int n,d;

    n=d=2048;

    //Creating dummy Q,K,Vs in CPP   --     Can generalise by reading from a file
    Q = (float*)malloc(sizeof(float)*n*d);
    K = (float*)malloc(sizeof(float)*n*d);
    V = (float*)malloc(sizeof(float)*n*d);
    S = (float*)malloc(sizeof(float)*n*n);
    O = (float*)malloc(sizeof(float)*n*d);

    //Initializing with values
    for(int i=0;i<n*d;i++){
        Q[i] = 1.0f;
        K[i] = 1.0f;
        V[i] = 1.0f;
    }

    //Creating memory in device
    cudaMalloc(&dQ, sizeof(float)*n*d);
    cudaMalloc(&dK, sizeof(float)*n*d);
    cudaMalloc(&dV, sizeof(float)*n*d);
    cudaMalloc(&dS, sizeof(float)*n*n);
    cudaMalloc(&dO, sizeof(float)*n*d);

    //Copying data from host to the device
    cudaMemcpy(dQ, Q, sizeof(float)*n*d, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeof(float)*n*d, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeof(float)*n*d, cudaMemcpyHostToDevice);

    //Configuring and launching kernels
    dim3 block(TILE,TILE);
    dim3 grid_qk((n+TILE-1)/TILE, (n+TILE-1)/TILE);

    //The QKt multiplication
    tiled_matmul_qk<<<grid_qk, block>>>(dQ, dK, dS, n, d);

    //Softmax calculation
    //the shared mem size argument -- 3rd position
    softmax<<<n, 256, 256*sizeof(float)>>>(dS, n);

    //The PV multiplication
    dim3 grid_out((d+TILE-1)/TILE, (n+TILE-1)/TILE);
    tiled_matmul_pv<<<grid_out, block>>>(dS, dV, dO, n, d);

    //Copying data from device to the host
    cudaMemcpy(S, dS, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(O, dO, sizeof(float)*n*d, cudaMemcpyDeviceToHost);

    std::cout<<"\n\nIntermediate S matrix:\n";
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            std::cout<<S[i*n + j]<<" ";
        }
        std::cout<<"\n";
    }
    
    std::cout<<"\n\nOutput O matrix";
    for(int i=0;i<n;i++){
        for(int j=0;j<d;j++){
            std::cout<<O[i*d + j]<<" ";
        }
        std::cout<<"\n";
    }
    
    //Freeing up device AND host memory
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dS);cudaFree(dO);
    free(Q);free(K);free(V);free(S);free(O);

    return 0;
}

/*
CUDA intuitions
    Decide upon the kernel configuration
    Generalisable grid, block, thread dimensions
    revisit matrix multiplication intution -- flat array tricks
        index = row * width + col
    Using threads to sweep through a row
        the key being that fewer threads can always get the work done 
*/