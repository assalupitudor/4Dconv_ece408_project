#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k14d(i3, i2, i1, i0) const_k1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define X_unroll(i2, i1, i0) x_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]


//baseline
__global__ void conv_forward_kernel(float*  y, const float*  x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    // add star slash if uncommented*/

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    //basic convilution implementation
    // Insert your GPU convolution kernel code here
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    if (h<H_out && w<W_out) {
       float val = 0.0;
       //#pragma unroll 2
       for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++){
                    val += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = val;
    }
    

}

//Shared Memory + tuning with restrict and loop unrolling + kernel values in constant memory. All three optimizations combined.
__constant__ float const_k1[3136];
__global__ void conv_forward_kernel1(float* __restrict__ y, const float* __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K) 
{
    

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y; int bx = blockIdx.x; int by = blockIdx.y;
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    const int h = blockIdx.z / W_grid * TILE_WIDTH + ty;
    const int w = blockIdx.z % W_grid * TILE_WIDTH + tx;
    int tile_xstart =  blockIdx.z % W_grid * TILE_WIDTH;
    int tile_ystart =  blockIdx.z / W_grid * TILE_WIDTH;
    int tile_xend = tile_xstart + TILE_WIDTH;
    int tile_yend = tile_ystart + TILE_WIDTH;

    int shift_val = K/2;
    float val = 0.0;
    #pragma unroll 2
    for (int c = 0; c < C; c++) {
        // load into the shared memory matrix
        if (h<H_out && w<W_out) {
            tile[ty][tx] = x4d(bx, c, h + shift_val, w + shift_val);
        } else {
            tile[ty][tx] = 0;
        }
        __syncthreads();

        int input_xstart =  tile_xstart + shift_val;
        int input_ystart = tile_ystart + shift_val;
        int input_xend = tile_xend + shift_val;
        int input_yend = tile_yend + shift_val;

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                // shift to input coordinate and then start from top left
                int input_xglobal = ((w - shift_val) + shift_val) + i;
                int input_yglobal = ((h - shift_val) + shift_val) + j; 
                int input_xshared = i + tx - shift_val;
                int input_yshared = j + ty - shift_val;
                if (input_xglobal < W && input_yglobal < H) {
                    int min_x = (input_xend < W_out + shift_val) ? input_xend : W_out + shift_val;
                    int min_y = (input_yend < H_out + shift_val) ? input_yend : H_out + shift_val;

                    //shared memory
                    if (input_xglobal >= input_xstart && input_xglobal < min_x && input_yglobal >= input_ystart && input_yglobal < min_y) {
                        val += tile[input_yshared][input_xshared] * k14d(by, c, j, i);
                    } else {
                        val += x4d(bx, c, input_yglobal, input_xglobal) * k14d(by, c, j, i);
                    }

                }
            }
        }
        __syncthreads();
    }

    if (h<H_out && w<W_out) {y4d(bx, by, h, w) = val;}

    

}


////Shared Memory + tuning with restrict and loop unrolling. NOTE comment out restrict and pragma unroll 2 to just get the shared memory optimization.
__global__ void conv_forward_kernel2(float* __restrict__  y, const float* __restrict__ x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) 
{
    


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y; int bx = blockIdx.x; int by = blockIdx.y;
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    const int h = blockIdx.z / W_grid * TILE_WIDTH + ty;
    const int w = blockIdx.z % W_grid * TILE_WIDTH + tx;
    int tile_xstart =  blockIdx.z % W_grid * TILE_WIDTH;
    int tile_ystart =  blockIdx.z / W_grid * TILE_WIDTH;
    int tile_xend = tile_xstart + TILE_WIDTH;
    int tile_yend = tile_ystart + TILE_WIDTH;

    int shift_val = K/2;
    float val = 0.0;
    #pragma unroll 2
    for (int c = 0; c < C; c++) {
        // load into the shared memory matrix
        if (h<H_out && w<W_out) {
            tile[ty][tx] = x4d(bx, c, h + shift_val, w + shift_val);
        } else {
            tile[ty][tx] = 0;
        }
        __syncthreads();

        int input_xstart =  tile_xstart + shift_val;
        int input_ystart = tile_ystart + shift_val;
        int input_xend = tile_xend + shift_val;
        int input_yend = tile_yend + shift_val;

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                // shift to input coordinate and then start from top left
                int input_xglobal = ((w - shift_val) + shift_val) + i;
                int input_yglobal = ((h - shift_val) + shift_val) + j; 
                int input_xshared = i + tx - shift_val;
                int input_yshared = j + ty - shift_val;
                if (input_xglobal < W && input_yglobal < H) {
                    int min_x = (input_xend < W_out + shift_val) ? input_xend : W_out + shift_val;
                    int min_y = (input_yend < H_out + shift_val) ? input_yend : H_out + shift_val;

                    //shared memory
                    if (input_xglobal >= input_xstart && input_xglobal < min_x && input_yglobal >= input_ystart && input_yglobal < min_y) {
                        val += tile[input_yshared][input_xshared] * k4d(by, c, j, i);
                    } else {
                        val += x4d(bx, c, input_yglobal, input_xglobal) * k4d(by, c, j, i);
                    }

                }
            }
        }
        __syncthreads();
    }

    if (h<H_out && w<W_out) {y4d(bx, by, h, w) = val;}

    
}


__global__ void unroll_gpu(const int C, const int H, const int W, const int K, const float* x, float* x_unroll, const int B, int B_old)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int c, s, h_out, w_out, w_unroll, h_unroll, w_base, p, q;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;
	if (t < C * W_unroll && b < (B-B_old)) {
		c = t / W_unroll;
		s = t % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		w_unroll = h_out * W_out + w_out;
		w_base = c * K * K;
	
		for(p = 0; p < K; p++){
			for(q = 0; q < K; q++) {
				h_unroll = w_base + p * K + q; 
				X_unroll(b, h_unroll, w_unroll) = x4d(b+B_old, c, h_out + p, w_out + q);				
			}
		}
	
	}		
}

__global__ void shared_forward_kernel(float*  y, const float*  x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int B_old)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int batch = blockIdx.z * blockDim.z + threadIdx.z ;

    int numARows = M;
	int numAColumns = C*K*K; 
	int numBRows = C*K*K;
	int numBColumns = W_unroll;
	int numCRows = numARows;
    int numCColumns = numBColumns;

    

    float Pvalue = 0;

    if (batch<B) {
        for (int i = 0; i < ceil((1.0*numAColumns)/((float)TILE_WIDTH)); i++) {
            if ((Row< numARows) && (i*TILE_WIDTH+tx)< numAColumns) {
                Mds[ty][tx] = k[Row*numAColumns + i*TILE_WIDTH + tx];
            } else {
                Mds[ty][tx] = 0;
            }
            if ((i*TILE_WIDTH+ty)<numBRows && Col<numBColumns) {
                Nds[ty][tx] = x[(batch) * ( numBColumns * numBRows ) + (i*TILE_WIDTH+ty) * numBColumns + Col];
            } else {
                Nds[ty][tx] = 0;
            }
            __syncthreads();
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();

        }
        if ((Row<numCRows) && (Col<numCColumns) && batch < B-B_old) {
            y[(batch+B_old)*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue;
        } 
    }
    
}



//k in constant memory 
__global__ void shared_forward_kernel_const_k(float*  y, const float*  x, /*const float *k,*/ const int B, const int M, const int C, const int H, const int W, const int K, int B_old)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int batch = blockIdx.z * blockDim.z + threadIdx.z ;

    int numARows = M;
	int numAColumns = C*K*K; 
	int numBRows = C*K*K;
	int numBColumns = W_unroll;
	int numCRows = numARows;
    int numCColumns = numBColumns;

    

    float Pvalue = 0;

    if (batch<B) {
        for (int i = 0; i < ceil((1.0*numAColumns)/((float)TILE_WIDTH)); i++) {
            if ((Row< numARows) && (i*TILE_WIDTH+tx)< numAColumns) {
                Mds[ty][tx] = const_k1[Row*numAColumns + i*TILE_WIDTH + tx];
            } else {
                Mds[ty][tx] = 0;
            }
            if ((i*TILE_WIDTH+ty)<numBRows && Col<numBColumns) {
                Nds[ty][tx] = x[(batch) * ( numBColumns * numBRows ) + (i*TILE_WIDTH+ty) * numBColumns + Col];
            } else {
                Nds[ty][tx] = 0;
            }
            __syncthreads();
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();

        }
        if ((Row<numCRows) && (Col<numCColumns) && batch < B-B_old) {
            y[(batch+B_old)*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue;
        } 
    }
    
}





//restrict plus loop unrolling
__global__ void shared_restrict_forward_kernel(float* __restrict__ y, const float* __restrict__ x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int B_old)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int batch = blockIdx.z * blockDim.z + threadIdx.z ;

    int numARows = M;
	int numAColumns = C*K*K; 
	int numBRows = C*K*K;
	int numBColumns = W_unroll;
	int numCRows = numARows;
    int numCColumns = numBColumns;

    

    float Pvalue = 0;

    if (batch<B) {
        #pragma unroll 2
        for (int i = 0; i < ceil((1.0*numAColumns)/((float)TILE_WIDTH)); i++) {
            if ((Row< numARows) && (i*TILE_WIDTH+tx)< numAColumns) {
                Mds[ty][tx] = k[Row*numAColumns + i*TILE_WIDTH + tx];
            } else {
                Mds[ty][tx] = 0;
            }
            if ((i*TILE_WIDTH+ty)<numBRows && Col<numBColumns) {
                Nds[ty][tx] = x[(batch) * ( numBColumns * numBRows ) + (i*TILE_WIDTH+ty) * numBColumns + Col];
            } else {
                Nds[ty][tx] = 0;
            }
            __syncthreads();
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();

        }
        if ((Row<numCRows) && (Col<numCColumns) && batch < B-B_old) {
            y[(batch+B_old)*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue;
        } 
    }
    
}


/*
__host__ void unroll_cpu(int B, int C, int H, int W, int K, float* x, float* x_unroll) 
{
    #define hostx4d(i3, i2, i1, i0) host_x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define X_unroll(i2, i1, i0) x_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]
    float* x_unroll = new float[B * (C * K * K * H_out * W_out)];	
    printf("reached def unroll \n");
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            int w_base = c*K*K;
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    for (int h = 0; h < H_out; h++) {
                        for (int w = 0; w < W_out; w++) {
                            int h_unroll = w_base + p*K + q;
                            int w_unroll = h*W_out +w;
                            X_unroll(b, h_unroll, w_unroll) = hostx4d(b, c, h+p, w+q);
                        }
                    }
                }
            }
        }
    }
    #undef hostx4d
    #undef X_unroll 
}
*/

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    cudaMalloc(device_y_ptr, B*M*H_out*W_out*sizeof(float));
    //printf("reached y alloc \n");
    //cudaMalloc(device_x_ptr, B*(C*K*K*H_out*W_out)*sizeof(float));
    cudaMalloc(device_x_ptr, B*C*H*W*sizeof(float));
    //printf("reached x alloc \n");
    cudaMalloc(device_k_ptr, M*C*K*K*sizeof(float));
    //printf("reached k alloc \n");

    

    cudaMemcpy(*device_x_ptr, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_x_ptr, x_unroll, B*(C*K*K*H_out*W_out)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    
    //cudaMemcpyToSymbol(const_k1, host_k, M*C*K*K*sizeof(float));

    
    //printf("%d \n", B);
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    

    /*
    Use this to get MCK values and take the larger ones to make the constant memory kernel const_k1
    printf("%d \n", M);
    printf("%d \n", C);
    printf("%d \n", K);
    B=10000
    */
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int H_grid = ceil((float)H_out / TILE_WIDTH);
    int Z = H_grid*W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    int numARows = M;
	int numAColumns = C*K*K; 
	int numBRows = C*K*K;
	int numBColumns = H_out * W_out;
	int numCRows = numARows;
    int numCColumns = numBColumns;
    
    int B_half = B/2;
    float *x_unroll;

    dim3 unroll_grid(ceil((float)1.0*C*H_out*W_out/TILE_WIDTH), ceil((float)1.0*B_half/TILE_WIDTH), 1);
    dim3 unroll_block(TILE_WIDTH, TILE_WIDTH, 1);

    dim3 matrix_block(TILE_WIDTH,TILE_WIDTH,1);
    dim3 matrix_grid(ceil((float) numCColumns/TILE_WIDTH), ceil((float)numCRows/TILE_WIDTH), B_half);
/*
    dim3 fusion_grid(unroll_grid.x,unroll_grid.y, B_half);
	dim3 fusion_block(TILE_WIDTH,TILE_WIDTH, 1);

    cudaMalloc((void **)&x_unroll, B_half * K * K * C * H_out* W_out*sizeof(float));
    fusion_kernel<<<fusion_grid, fusion_block>>>(device_y, device_x, x_unroll, device_k, B, M, C, H, W, K, 0);
    cudaFree(x_unroll);

    cudaMalloc((void **)&x_unroll, B_half * K * K * C * H_out* W_out*sizeof(float));
    fusion_kernel<<<fusion_grid, fusion_block>>>(device_y, device_x, x_unroll, device_k, B, M, C, H, W, K, B_half);
    cudaFree(x_unroll);
*/


    
    cudaMalloc((void **)&x_unroll, B_half * K * K * C * H_out* W_out*sizeof(float));
    unroll_gpu<<<unroll_grid, unroll_block>>>(C, H, W, K, device_x, x_unroll, B_half, 0);
    shared_forward_kernel<<<matrix_grid, matrix_block>>>(device_y, x_unroll, device_k, B, M, C, H, W, K, 0);
    //shared_restrict_forward_kernel<<<matrix_grid, matrix_block>>>(device_y, x_unroll, device_k, B, M, C, H, W, K, 0);
    //shared_forward_kernel_const_k<<<matrix_grid, matrix_block>>>(device_y, x_unroll, B, M, C, H, W, K, 0);
    cudaFree(x_unroll);

    cudaMalloc((void **)&x_unroll, B_half * K * K * C * H_out* W_out*sizeof(float));
    unroll_gpu<<<unroll_grid, unroll_block>>>(C, H, W, K, device_x, x_unroll, B, B_half);
    shared_forward_kernel<<<matrix_grid, matrix_block>>>(device_y, x_unroll, device_k, B, M, C, H, W, K, B_half);
    //shared_restrict_forward_kernel<<<matrix_grid, matrix_block>>>(device_y, x_unroll, device_k, B, M, C, H, W, K, B_half);
    //shared_forward_kernel_const_k<<<matrix_grid, matrix_block>>>(device_y, x_unroll, B, M, C, H, W, K, B_half);
    cudaFree(x_unroll);



    //conv_forward_kernel1<<<gridDim, blockDim>>>(device_y, device_x, B, M, C, H, W, K);
    //conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    
    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, B*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
    


}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}


#undef y4d
#undef x4d
#undef k4d
#undef k14d
#undef X_unroll








/*
__global__ void matrixMultiplyShared(const float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns
    int bacth) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int layer_number = blockIdx.z * blockDim.z + threadIdx.z;
    float Pvalue = 0;

    for (int i = 0; i < ceil((1.0*numAColumns)/((float)TILE_WIDTH)); i++) {
        if ((Row< numARows) && (i*TILE_WIDTH+tx)< numAColumns) {
            Mds[ty][tx] = A[Row*numAColumns + i*TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0;
        }
        if ((i*TILE_WIDTH+ty)<numBRows && Col<numBColumns) {
            Nds[ty][tx] = B[layer_number*numBColumns*numBRows + (i*TILE_WIDTH + ty)*numBColumns + Col];
        } else {
            Nds[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if ((Row<numCRows) && (Col<numCColumns) && layer_number < batch) {
        C[(layer_number)*numCColumns*numCRows + Row*numCColumns + Col] = Pvalue;
    } 

}
*/