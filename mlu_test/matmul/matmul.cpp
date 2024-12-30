#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "cnrt.h"
#include "cnnl.h"

using namespace std;
using namespace std::chrono;

void test_matmul(int batch, int m, int k, int n) {
	std::vector<float> host_A(batch*m*k, 1.0f);
	std::vector<float> host_B(batch*n*k, 1.0f);
	std::vector<float> host_C(batch*m*n);
	
	cnInit(0);
	// printf("first ele before:%f\n", host_C[0]);

	void *d_A, *d_B, *d_C;
	cnrtMalloc(&d_A, sizeof(float) * host_A.size());
	cnrtMalloc(&d_B, sizeof(float) * host_B.size());
	cnrtMalloc(&d_C, sizeof(float) * host_C.size());

	// memory copy
	cnrtMemcpy(d_A, host_A.data(), sizeof(float)*host_A.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);
	cnrtMemcpy(d_B, host_B.data(), sizeof(float)*host_B.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);

	cnnlTensorDescriptor_t A_desc = nullptr;
	cnnlTensorDescriptor_t B_desc = nullptr;
	cnnlTensorDescriptor_t C_desc = nullptr;
	cnnlCreateTensorDescriptor(&A_desc);
	cnnlCreateTensorDescriptor(&B_desc);
	cnnlCreateTensorDescriptor(&C_desc);

	int A_dim[3] = {batch, m, k};
	int B_dim[3] = {batch, k, n};
	int C_dim[3] = {batch, m, n};

	cnnlSetTensorDescriptor(A_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, A_dim);
	cnnlSetTensorDescriptor(B_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, B_dim);
	cnnlSetTensorDescriptor(C_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, C_dim);

	cnnlHandle_t handle = nullptr;
	cnnlCreate(&handle);

	// utils
	cnnlMatMulDescriptor_t mm_desc;
	cnnlMatMulAlgo_t algo;
	void* workspace = nullptr;
	size_t workspace_size = 0;
	cnnlDataType_t compute_dtype = CNNL_DTYPE_FLOAT;

	// attr
	cnnlMatMulDescCreate(&mm_desc);
	cnnlSetMatMulDescAttr(mm_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE, &compute_dtype, sizeof(cnnlDataType_t));

	int requested_algo_count = 1;
	cnnlMatMulHeuristicResult_t result_array;
	cnnlCreateMatMulHeuristicResult(&result_array);
	int return_algo_count;
	cnnlGetBatchMatMulAlgoHeuristic(handle, mm_desc, A_desc, B_desc, C_desc, NULL, requested_algo_count,
			&result_array, &return_algo_count);
	// printf("return_algo_count: %d\n", return_algo_count);

	cnnlMatMulAlgoCreate(&algo);
	cnnlGetBatchMatMulHeuristicResult(result_array, algo, &workspace_size);
	if(workspace_size != 0){
		printf("ws size: %ld\n", workspace_size);
		cnrtMalloc(&workspace, workspace_size);
	}
	
	// matmul init
	float alpha =1.0f;
	float beta = 0.0f;
	vector<double> result;
	for(int i=0; i<20; i++){
		steady_clock::time_point start = std::chrono::steady_clock::now();
		// do computation
		cnnlStatus_t status = cnnlBatchMatMulBCast_v2(handle, mm_desc, algo, &alpha, A_desc, d_A, B_desc, d_B, &beta, C_desc, d_C,
				workspace, workspace_size);
		// synchronize
		cnrtSyncDevice();
		if(status != CNNL_STATUS_SUCCESS)
			printf("error, error code: %d\n", status);

		steady_clock::time_point end = std::chrono::steady_clock::now();
		duration<double> elapse_time = duration_cast<std::chrono::microseconds>(end - start);
		result.push_back(elapse_time.count());
	}
	std::sort(result.begin(), result.end());
	// double mid_time = result.at(result.size()/2);
	double mid_time = result.at(0);
	

	double flops = (double)batch*(double)m*(double)k*(double)n*2 + (double)batch*(double)m*(double)n;
	// printf("tflops: %.5lf\n", flops);
	double GFLOPS = flops / 1e12 / mid_time;
	printf("TFLOPS:%.4lf \n", GFLOPS);
	
	// result copy
	cnrtMemcpy(host_C.data(), d_C, sizeof(float)*host_C.size(), CNRT_MEM_TRANS_DIR_DEV2HOST);
	// printf("first ele after:%f\n", host_C[0]);
	// cnrtDeviceReset();
}

int tcase[24] = {
	1, 1, 2048, 2048,
	16, 20, 32, 20,
	2, 20, 256, 768,
	2, 20, 256, 1024,
	128, 1, 128, 58,
	128, 1, 58, 128
};

// cncc matmul1.cpp -o mm1 -lcnnl -lcnrt -lcndrv -std=c++11 -fPIC -lstdc++ -lm -I${NEUWARE_HOME}/include -L${NEUWARE_HOME}/lib64
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
		test_matmul(batch, m, k, n);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			int batch = tcase[i*4];
			int m = tcase[i*4+1];
			int k = tcase[i*4+2];
			int n = tcase[i*4+3];

			printf("---------------------------------------------------\n");
			printf("matmul test case: %d\n", i);
			test_matmul(batch, m, k, n);
		}

	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
}