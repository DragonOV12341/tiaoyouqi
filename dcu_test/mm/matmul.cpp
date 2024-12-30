// hipcc  -DROCM_USE_FLOAT16 -o matmul_rocblas matmul_rocblas.cpp -lrocblas

#include <iostream>
#include <vector>
#include <cmath>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm> // For std::sort

// Check ROCm API status
#define CHECK_ROCBLAS_STATUS(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

int tcase[24] = {
	1, 1, 2048, 2048,
	16, 20, 32, 20,
	2, 20, 256, 768,
	2, 20, 256, 1024,
	128, 1, 128, 58,
	128, 1, 58, 128
};

void RunMatmulBenchmark(int batch, int m, int k, int n) {
    const int M = m;
    const int N = n;
    const int K = k;

    // Leading dimensions and stride
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    // Scalar multipliers
    float alpha = 1.0f;
    float beta = 1.0f;

    // Host matrices
    float *h_A, *h_B, *h_C, *h_D;
    h_A = (float *)malloc(sizeof(float) * M * K * batch);
    h_B = (float *)malloc(sizeof(float) * K * N * batch);
    h_C = (float *)malloc(sizeof(float) * M * N * batch);

    float *C_float = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K * batch; i++) {
        h_A[i] = float(1.0f);
    }
    for (int i = 0; i < K * N * batch; i++) {
        h_B[i] = float(1.0f);
    }
    for (int i = 0; i < M * N * batch; i++) {
        h_C[i] = float(0.0f);
    }

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // Allocate device memory
    hipMalloc((void**)(&d_A), M * K * batch * sizeof(float));
    hipMalloc((void**)(&d_B), K * N * batch * sizeof(float));
    hipMalloc((void**)(&d_C), M * N * batch * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_A, h_A, M * K * batch * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * batch * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C, M * N * batch * sizeof(float), hipMemcpyHostToDevice);

    float **h_AA, **h_BA, **h_CA;
    h_AA = (float **)malloc(sizeof(float*) * batch);
    h_BA = (float **)malloc(sizeof(float*) * batch);
    h_CA = (float **)malloc(sizeof(float*) * batch);
    for(int i=0; i<batch; i++) {
        h_AA[i] = d_A + M * K * i;
        h_BA[i] = d_B + K * N * i;
        h_CA[i] = d_C + M * N * i;
    }

    float **d_AA, **d_BA, **d_CA;
    hipMalloc((void**)(&d_AA), batch * sizeof(float*));
    hipMalloc((void**)(&d_BA), batch * sizeof(float*));
    hipMalloc((void**)(&d_CA), batch * sizeof(float*));

    hipMemcpy(d_AA, h_AA, batch * sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(d_BA, h_BA, batch * sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(d_CA, h_CA, batch * sizeof(float*), hipMemcpyHostToDevice);

    // HIP events for timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // rocblas_datatype_f32_r
    // rocblas_gemm_batched_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, 
    //         rocblas_int m, rocblas_int n, rocblas_int k, const void *alpha, const void *a, rocblas_datatype a_type, 
    //         rocblas_int lda, const void *b, rocblas_datatype b_type, rocblas_int ldb, const void *beta, const void *c, 
    //         rocblas_datatype c_type, rocblas_int ldc, void *d, rocblas_datatype d_type, rocblas_int ldd, 
    //         rocblas_int batch_count, rocblas_datatype compute_type, rocblas_gemm_algo algo, int32_t solution_index, 
    //         uint32_t flags)

    // Warmup
    CHECK_ROCBLAS_STATUS(rocblas_gemm_batched_ex(
                handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha,
                d_AA, rocblas_datatype_f32_r, lda, d_BA, rocblas_datatype_f32_r, ldb, &beta,
                d_CA, rocblas_datatype_f32_r, ldc, d_CA, rocblas_datatype_f32_r, ldc, 
                batch, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, -1, 0
            ));

    hipDeviceSynchronize();

    // Measure 10 times
    std::vector<float> elapsed_times(10);
    for (int i = 0; i < 10; ++i) {
        // Record start event
        hipEventRecord(start, 0);

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        CHECK_ROCBLAS_STATUS(rocblas_gemm_batched_ex(
                    handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha,
                    d_AA, rocblas_datatype_f32_r, lda, d_BA, rocblas_datatype_f32_r, ldb, &beta,
                    d_CA, rocblas_datatype_f32_r, ldc, d_CA, rocblas_datatype_f32_r, ldc, 
                    batch, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, -1, 0
                ));

        // Record stop event
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        // Calculate elapsed time
        hipEventElapsedTime(&elapsed_times[i], start, stop);
    }

    // Calculate median time using sort
    std::sort(elapsed_times.begin(), elapsed_times.end());
    float median_time_ms = elapsed_times[0];
    // float median_time_ms = elapsed_times[elapsed_times.size() / 2];

    // Print median execution time
    std::cout << "Median elapsed time: " << median_time_ms << " ms" << std::endl;

	double flops = (double)batch*(double)m*(double)k*(double)n*2 + (double)batch*(double)m*(double)n;
	// printf("tflops: %.5lf\n", flops);
	double GFLOPS = flops / 1e12 / (median_time_ms / 1000);
	printf("TFLOPS:%.4lf \n", GFLOPS);

    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_AA);
    hipFree(d_BA);
    hipFree(d_CA);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] C_float;
    free(h_AA);
    free(h_BA);
    free(h_CA);
    rocblas_destroy_handle(handle);
    hipEventDestroy(start);
    hipEventDestroy(stop);
}

int main(int argc, char* argv[]) {
	int caseidx = -1;
	if(argc == 2)
		caseidx = std::atoi(argv[1]);
	
	if(caseidx >=0 && caseidx <= 5){
		int batch = tcase[caseidx*4];
		int m = tcase[caseidx*4+1];
		int k = tcase[caseidx*4+2];
		int n = tcase[caseidx*4+3];

		// printf("test case: %d, batch-m-k-n: %d, %d, %d, %d\n", caseidx, batch, m, k, n);
		printf("matmul test case: %d\n", caseidx);
		RunMatmulBenchmark(batch, m, k, n);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			int batch = tcase[i*4];
			int m = tcase[i*4+1];
			int k = tcase[i*4+2];
			int n = tcase[i*4+3];

			printf("---------------------------------------------------\n");
			printf("matmul test case: %d\n", i);
			RunMatmulBenchmark(batch, m, k, n);
		}

	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
}