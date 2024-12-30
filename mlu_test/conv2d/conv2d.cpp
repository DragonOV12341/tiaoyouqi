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
    1, 64, 64, 112, 56, 3, 2, 1, 1,
    1, 64, 128, 56, 28, 3, 2, 1, 1,
    1, 128, 256, 28, 14, 3, 2, 1, 1,
    1, 320, 320, 64, 64, 3, 1, 1, 1,
    1, 640, 640, 32, 32, 3, 1, 1, 1,
    1, 1280, 1280, 16, 16, 3, 1, 1, 1,
};

void test_conv2d(int idx) {
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
	std::vector<float> host_weight(inC*outC*kernel_size*kernel_size, 1.0f);
    std::vector<float> host_bias(outC, 1.0f);
	std::vector<float> host_output(batch*outC*outHnW*outHnW);
	
	cnInit(0);
	// printf("first ele before:%f\n", host_C[0]);

	void *d_in, *d_wei, *d_bias, *d_out;
	cnrtMalloc(&d_in, sizeof(float) * host_input.size());
	cnrtMalloc(&d_wei, sizeof(float) * host_weight.size());
    cnrtMalloc(&d_bias, sizeof(float) * host_bias.size());
	cnrtMalloc(&d_out, sizeof(float) * host_output.size());

	// memory copy
	cnrtMemcpy(d_in, host_input.data(), sizeof(float)*host_input.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);
	cnrtMemcpy(d_wei, host_weight.data(), sizeof(float)*host_weight.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(d_bias, host_bias.data(), sizeof(float)*host_bias.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);

	cnnlTensorDescriptor_t in_desc = nullptr;
	cnnlTensorDescriptor_t wei_desc = nullptr;
    cnnlTensorDescriptor_t bias_desc = nullptr;
	cnnlTensorDescriptor_t out_desc = nullptr;
	cnnlCreateTensorDescriptor(&in_desc);
	cnnlCreateTensorDescriptor(&wei_desc);
    cnnlCreateTensorDescriptor(&bias_desc);
	cnnlCreateTensorDescriptor(&out_desc);

	int in_dim[4] = {batch, inHnW, inHnW, inC};
	int wei_dim[4] = {outC, kernel_size, kernel_size, inC};
    int bias_dim[4] = {outC};
	int out_dim[4] = {batch, outHnW, outHnW, outC};

	cnnlSetTensorDescriptor(in_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, in_dim);
	cnnlSetTensorDescriptor(wei_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, wei_dim);
    cnnlSetTensorDescriptor(bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, bias_dim);
	cnnlSetTensorDescriptor(out_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, out_dim);

	cnnlHandle_t handle = nullptr;
	cnnlCreate(&handle);

	// utils
	cnnlConvolutionDescriptor_t conv_desc;
	cnnlConvolutionForwardAlgo_t algo;
 	float alpha = 1.0f;
	float beta = 0.0f;      
	void* workspace = nullptr;
	size_t workspace_size = 0;
	cnnlDataType_t compute_dtype = CNNL_DTYPE_FLOAT;

	cnnlCreateConvolutionDescriptor(&conv_desc);
    int pad[4] = {padding_, padding_, padding_, padding_};
    int stride[4] = {stride_, stride_, stride_, stride_};
    int dilation[4] = {dilation_, dilation_, dilation_, dilation_};
	cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, 1, compute_dtype);
 
    // cnnlGetConvolutionForwardAlgorithm(handle, conv_desc, in_desc, wei_desc, out_desc, 
    //         CNNL_CONVOLUTION_FWD_FASTEST, &algo);
    algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    cnnlGetConvolutionForwardWorkspaceSize(handle, in_desc, wei_desc, out_desc, bias_desc, 
            conv_desc, algo, &workspace_size);
	if(workspace_size != 0){
		printf("ws size: %ld\n", workspace_size);
		cnrtMalloc(&workspace, workspace_size);
	}

    // cnnlConvolutionCastMode_t castmode = CNNL_NO_QUANTIZE;
    // int requested_algo_count = 8;
    // int returned_algo_count;
    // cnnlConvolutionFwdAlgoPerf_t perfs[8];
    // cnnlFindConvolutionForwardAlgorithm(handle, conv_desc, castmode, &alpha, in_desc,
    //         d_in, wei_desc, d_wei, bias_desc, d_bias, &beta, out_desc, d_out, 8, perfs,
    //         returned_algo_count, )

	vector<double> result;
	for(int i=0; i<20; i++){
		steady_clock::time_point start = std::chrono::steady_clock::now();
		// do computation
		cnnlStatus_t status = cnnlConvolutionForward(handle, conv_desc, algo, NULL, in_desc, d_in, 
                wei_desc, d_wei, bias_desc, d_bias, workspace, workspace_size, NULL, out_desc, d_out);
        
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
            kernel_size*inC*2 + (double)batch*(double)outHnW*(double)outHnW*(double)outC;    
    
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
		printf("test case: %d\n", caseidx);
		test_conv2d(caseidx);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			printf("---------------------------------------------------\n");
			printf("conv2d test case: %d\n", i);
			test_conv2d(i);
		}
	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
}