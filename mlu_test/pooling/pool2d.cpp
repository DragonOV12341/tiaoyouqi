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

std::vector<int> tcase = {
    1, 96, 96, 55, 27, 3, 2, 0, 1,
    1, 256, 256, 27, 13, 3, 2, 0, 1,
    1, 256, 256, 13, 6, 3, 2, 0, 1,
    1, 64, 64, 112, 56, 3, 2, 0, 1,
    1, 2048, 2048, 7, 1, 7, 1, 0, 1,
    1, 256, 256, 20, 20, 5, 1, 2, 1,
};

void test_pool2d(int idx) {
    int batch = tcase[idx*9 + 0];
    int inC = tcase[idx*9 + 1];
    int outC = tcase[idx*9 + 2];
    int inHnW = tcase[idx*9 + 3];
    int outHnW = tcase[idx*9 + 4];
    int kernel_size = tcase[idx*9 + 5];
    int stride_ = tcase[idx*9 + 6];
    int padding_ = tcase[idx*9 + 7];
    int dilation_ = tcase[idx*9 + 8];

	std::vector<float> host_input(batch*inC*inHnW*inHnW, 1.0f);
	std::vector<float> host_output(batch*outC*outHnW*outHnW);
	float* extra_in = (float*)malloc(sizeof(float)*host_input.size());

	cnInit(0);
	// printf("first ele before:%f\n", host_C[0]);

	void *d_in, *d_out, *d_extrain;
	cnrtMalloc(&d_in, sizeof(float) * host_input.size());
	cnrtMalloc(&d_extrain, sizeof(float) * host_input.size());
	cnrtMalloc(&d_out, sizeof(float) * host_output.size());

	// memory copy
	cnrtMemcpy(d_in, host_input.data(), sizeof(float)*host_input.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);

	cnnlTensorDescriptor_t in_desc = nullptr;
	cnnlTensorDescriptor_t out_desc = nullptr;
	cnnlCreateTensorDescriptor(&in_desc);
	cnnlCreateTensorDescriptor(&out_desc);

	int in_dim[4] = {batch, inHnW, inHnW, inC};
	int out_dim[4] = {batch, outHnW, outHnW, outC};

	cnnlSetTensorDescriptor(in_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, in_dim);
	cnnlSetTensorDescriptor(out_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, out_dim);

	cnnlHandle_t handle = nullptr;
	cnnlCreate(&handle);

	// utils
	cnnlPoolingDescriptor_t pool_desc;
	cnnlPoolingMode_t mode = CNNL_POOLING_MAX;
	if(idx == 4)
		mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
 	float alpha = 1.0f;
	float beta = 0.0f;
	void* workspace = nullptr;
	size_t workspace_size = 0;
	cnnlDataType_t compute_dtype = CNNL_DTYPE_FLOAT;

	cnnlCreatePoolingDescriptor(&pool_desc);
    int pad[4] = {padding_, padding_, padding_, padding_};
    int stride[2] = {stride_, stride_};
    int dilation[2] = {dilation_, dilation_};
	cnnlSetPooling2dDescriptor_v2(pool_desc, mode, CNNL_NOT_PROPAGATE_NAN, kernel_size, 
			kernel_size, pad[0], pad[1], pad[2], pad[3], stride[0], stride[1],
			dilation[0], dilation[1], true);
	
    cnnlGetPoolingWorkspaceSize(handle, mode, outHnW, outHnW, &workspace_size);
	if(workspace_size != 0){
		printf("ws size: %ld\n", workspace_size);
		cnrtMalloc(&workspace, workspace_size);
	}
	
	cnnlInitPoolingExtraInput(handle, pool_desc, in_desc, out_desc, extra_in);
	cnrtMemcpy(d_extrain, extra_in, sizeof(float)*host_input.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);

	vector<double> result;
	for(int i=0; i<20; i++){
		steady_clock::time_point start = std::chrono::steady_clock::now();
		// do computation
		cnnlStatus_t status = cnnlPoolingForward_v2(handle, pool_desc, NULL, in_desc, d_in, 
                 NULL, d_extrain, out_desc, d_out, workspace, workspace_size);
        
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
	
	double flops = (double)batch*(double)outHnW*(double)outHnW*(double)outC*(double)kernel_size*
            kernel_size*inC;
    
	// printf("tflops: %.5lf\n", flops);
	double TFLOPS = flops / 1e12 / mid_time;
    printf("TFLOPS:%.4lf \n", TFLOPS);
	// printf("mid time: %.4lf ms, GFLOPS:%.4lf \n", mid_time*1000, TFLOPS);
	
	// result copy
	// cnrtMemcpy(host_C.data(), d_C, sizeof(float)*host_C.size(), CNRT_MEM_TRANS_DIR_DEV2HOST);
	// printf("first ele after:%f\n", host_C[0]);
	// cnrtDeviceReset();
}

// cncc conv2d.cpp -o conv2d -lcnnl -lcnrt -lcndrv -std=c++11 -fPIC -lstdc++ -lm -I${NEUWARE_HOME}/include -L${NEUWARE_HOME}/lib64
int main(int argc, char* argv[]) {
	int caseidx = -1;
	if(argc == 2)
		caseidx = std::atoi(argv[1]);
	
	if(caseidx >=0 && caseidx <= 5){
		// printf("test case: %d, batch-m-k-n: %d, %d, %d, %d\n", caseidx, batch, m, k, n);
		printf("pooling2d test case: %d\n", caseidx);
		test_pool2d(caseidx);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			printf("---------------------------------------------------\n");
			printf("pooling2d test case: %d\n", i);
			test_pool2d(i);
		}
	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
}