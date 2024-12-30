// hipcc pool_miopen.cpp -o pool_miopen -lMIOpen
#include <miopen/miopen.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <hip/hip_runtime.h>

// 定义一些帮助宏
#define MIOPEN_CALL(cmd)                                                      \
    {                                                                         \
        miopenStatus_t status = cmd;                                          \
        if (status != miopenStatusSuccess) {                                  \
            std::cerr << "MIOpen Error: " << miopenGetErrorString(status)     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

std::vector<int> tcase = {
    1, 96, 96, 55, 27, 3, 2, 0, 1,
    1, 256, 256, 27, 13, 3, 2, 0, 1,
    1, 256, 256, 13, 6, 3, 2, 0, 1,
    1, 224, 224, 112, 56, 3, 2, 0, 1,
    1, 1024, 1024, 7, 1, 7, 1, 0, 1,
    1, 512, 512, 20, 20, 5, 1, 2, 1,
};

void RunPoolingBenchmark(int idx) {
    int deviceId = 0;
    hipSetDevice(deviceId);

    // 验证当前设备
    // int currentDevice;
    // hipGetDevice(&currentDevice);
    // std::cout << "Current GPU: " << currentDevice << std::endl;

    // 初始化 MIOpen handle
    miopenHandle_t handle;
    MIOPEN_CALL(miopenCreate(&handle));

    // 配置输入张量
    int n = tcase[idx*9 + 0];
    int c = tcase[idx*9 + 1];
    // int outC = tcase[idx*9 + 2];
    int h = tcase[idx*9 + 3];
    int w = h;
    // int outHnW = tcase[idx*9 + 4];
    int kernel_h = tcase[idx*9 + 5];
    int kernel_w = kernel_h;
    int stride_h = tcase[idx*9 + 6];
    int stride_w = stride_h;
    int pad_h = tcase[idx*9 + 7];
    int pad_w = pad_h;

    // 初始化张量描述符
    miopenTensorDescriptor_t input_desc, output_desc;
    MIOPEN_CALL(miopenCreateTensorDescriptor(&input_desc));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&output_desc));

    MIOPEN_CALL(miopenSet4dTensorDescriptor(input_desc, miopenHalf, n, c, h, w));

    // 初始化池化描述符
    miopenPoolingDescriptor_t pool_desc;
    MIOPEN_CALL(miopenCreatePoolingDescriptor(&pool_desc));
    MIOPEN_CALL(miopenSet2dPoolingDescriptor(pool_desc, miopenPoolingAverage, kernel_h, kernel_w,
                                             pad_h, pad_w, stride_h, stride_w));

    // 计算输出张量的维度
    int out_n, out_c, out_h, out_w;
    MIOPEN_CALL(miopenGetPoolingForwardOutputDim(pool_desc, input_desc, &out_n, &out_c, &out_h, &out_w));
    MIOPEN_CALL(miopenSet4dTensorDescriptor(output_desc, miopenHalf, out_n, out_c, out_h, out_w));

    // std::cout << "Output Tensor Shape: ( out_n: " << out_n << ", out_c: " << out_c
    //           << ", out_h: " << out_h << ", out_w: " << out_w << ")" << std::endl;

    // 分配内存
    size_t input_size = n * c * h * w * sizeof(_Float16);
    size_t output_size = out_n * out_c * out_h * out_w * sizeof(_Float16);

    _Float16* h_input = (_Float16*)malloc(input_size);
    _Float16* h_output = (_Float16*)malloc(output_size);

    // 初始化数据
    memset(h_input, (_Float16)1, input_size);
    memset(h_output, (_Float16)0, output_size);

    _Float16 *input, *output;
    hipMalloc((void **)&input, input_size);
    hipMalloc((void **)&output, output_size);

    hipMemcpy(input, h_input, input_size, hipMemcpyHostToDevice);
    hipMemcpy(output, h_output, output_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    // 池化操作通常不需要额外 workspace
    size_t workspace_size = 0;
    void* workspace{nullptr};
    if (workspace_size > 0) {
        std::cout << "Workspace size: " << (workspace_size / 1048576.0) << "MB"
                  << std::endl;
        hipMalloc((void **)(&workspace), workspace_size);
    }

    // 池化前向计算
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    std::vector<float> elapsed_times(10);
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CALL(miopenPoolingForward(handle, pool_desc, &alpha, input_desc, input,
                                     &beta, output_desc, output, false, workspace, workspace_size));
    hipDeviceSynchronize();

    for (int i = 0; i < 10; ++i) {
        hipEventRecord(start, 0);
        MIOPEN_CALL(miopenPoolingForward(handle, pool_desc, &alpha, input_desc, input,
                                         &beta, output_desc, output, false, workspace, workspace_size));
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed_times[i], start, stop);
        // std::cout << "Repeat " << i << " Pooling forward executed successfully!" << std::endl;
    }

    std::sort(elapsed_times.begin(), elapsed_times.end());
    float median_time_ms = elapsed_times[0];
    // float median_time_ms = elapsed_times[elapsed_times.size() / 2];
    std::cout << "Median elapsed time: " << median_time_ms << " ms" << std::endl;

	double flops = (double)n*(double)out_w*(double)out_h*(double)out_c*(double)kernel_h*
            kernel_w*c;
	double TFLOPS = flops / 1e12 / (median_time_ms / 1000);
    printf("TFLOPS:%.4lf \n", TFLOPS);

    // 释放资源
    hipFree(input);
    hipFree(output);
    if (workspace) {
        hipFree(workspace);
    }
    MIOPEN_CALL(miopenDestroyTensorDescriptor(input_desc));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(output_desc));
    MIOPEN_CALL(miopenDestroyPoolingDescriptor(pool_desc));
    MIOPEN_CALL(miopenDestroy(handle));

    free(h_input);
    free(h_output);
}

int main(int argc, char* argv[]) {
	int caseidx = -1;
	if(argc == 2)
		caseidx = std::atoi(argv[1]);
	
	if(caseidx >=0 && caseidx <= 5){
		// printf("test case: %d, batch-m-k-n: %d, %d, %d, %d\n", caseidx, batch, m, k, n);
		printf("pooling2d test case: %d\n", caseidx);
		RunPoolingBenchmark(caseidx);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			printf("---------------------------------------------------\n");
			printf("pooling2d test case: %d\n", i);
			RunPoolingBenchmark(i);
		}
	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
    return 0;
}
