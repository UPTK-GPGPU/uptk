#include <iostream>
#include <cuda_runtime_api.h> 
#include <UPTK_runtime_api.h> 

__global__ void vectorAdd(float *a, float *b, float *c, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10; 
    const int size = N * sizeof(float); 

    float *h_a = (float*)malloc(size); 
    float *h_b = (float*)malloc(size); 
    float *h_c = (float*)malloc(size); 

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);      
        h_b[i] = static_cast<float>(i * 2);  
    }

    float *d_a, *d_b, *d_c;
    UPTKMalloc(&d_a, size); 
    UPTKMalloc(&d_b, size); 
    UPTKMalloc(&d_c, size); 

    UPTKMemcpy(d_a, h_a, size, UPTKMemcpyHostToDevice);
    UPTKMemcpy(d_b, h_b, size, UPTKMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 


    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    UPTKDeviceSynchronize();

    UPTKMemcpy(h_c, d_c, size, UPTKMemcpyDeviceToHost);

    std::cout << "Result of vector addition:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    free(h_a); free(h_b); free(h_c);           
    UPTKFree(d_a); UPTKFree(d_b); UPTKFree(d_c); 

    return 0;
}
