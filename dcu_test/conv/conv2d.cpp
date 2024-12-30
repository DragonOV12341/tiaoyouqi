// hipcc conv_miopen_f16.cpp -o conv_miopen_f16 -lMIOpen
#include <miopen/miopen.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

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
    1, 64, 64, 112, 56, 3, 2, 1, 1,
    1, 64, 128, 56, 28, 3, 2, 1, 1,
    1, 128, 256, 28, 14, 3, 2, 1, 1,
    1, 160, 160, 64, 64, 3, 1, 1, 1,
    1, 240, 240, 32, 32, 3, 1, 1, 1,
    1, 320, 320, 16, 16, 3, 1, 1, 1,
};

void RunConv2DExample(int idx) {
    int deviceId = 0;
    hipSetDevice(deviceId);

    // 验证当前设备
    int currentDevice;
    hipGetDevice(&currentDevice);
    std::cout << "Current GPU: " << currentDevice << std::endl;
    // 初始化 MIOpen handle
    miopenHandle_t handle;
    MIOPEN_CALL(miopenCreate(&handle));

    // 配置输入张量
    int n = tcase[idx*9 + 0];
    int c = tcase[idx*9 + 1];
    int outC = tcase[idx*9 + 2];
    int h = tcase[idx*9 + 3]; int w = h;
    int outHnW = tcase[idx*9 + 4];
    int kernel_size = tcase[idx*9 + 5];
    int filter_n = outC, filter_c = c, filter_h = kernel_size, filter_w = kernel_size;
    int stride_h = tcase[idx*9 + 6]; int stride_w = stride_h;
    int pad_h = tcase[idx*9 + 7]; int pad_w = pad_h;
    int dilation_h = tcase[idx*9 + 8]; int dilation_w = dilation_h;
    int group = 1;

    // int n = 1, c = 1280, h = 16, w = 16; // 输入张量: (batch, channels, height, width)
    // // 卷积核: (output_channels, input_channels, kernel_h, kernel_w)
    // int filter_n = 1280, filter_c = 1280, filter_h = 3, filter_w = 3; 
    // int pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1; // 卷积参数
    // int group = 1;

    miopenConvolutionMode_t mode{miopenConvolution};
    // 初始化张量描述符
    miopenTensorDescriptor_t input_desc, filter_desc, output_desc;
    MIOPEN_CALL(miopenCreateTensorDescriptor(&input_desc));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&filter_desc));
    MIOPEN_CALL(miopenCreateTensorDescriptor(&output_desc));

    MIOPEN_CALL(miopenSet4dTensorDescriptor(input_desc, miopenHalf, n, c, h, w));
    MIOPEN_CALL(miopenSet4dTensorDescriptor(filter_desc, miopenHalf, filter_n, filter_c / group, filter_h, filter_w));
    // 初始化卷积描述符
    miopenConvolutionDescriptor_t conv_desc;
    MIOPEN_CALL(miopenCreateConvolutionDescriptor(&conv_desc));
    MIOPEN_CALL(miopenInitConvolutionDescriptor(conv_desc, mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w));
    if (group > 1) {
        MIOPEN_CALL(miopenSetConvolutionGroupCount(conv_desc, group));
    }

    // 计算输出张量的维度
    int out_n, out_c, out_h, out_w;
    MIOPEN_CALL(miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, filter_desc, &out_n, &out_c, &out_h, &out_w));
    MIOPEN_CALL(miopenSet4dTensorDescriptor(output_desc, miopenHalf, out_n, out_c, out_h, out_w));
    // std::cout << "Output Tensor Shape: ( out_n: " << out_n << ", out_c: " << out_c << ", out_h: " << out_h << ", out_w: " << out_w << ")" << std::endl;

    // 分配内存
    size_t input_size = n * c * h * w * sizeof(_Float16);
    size_t filter_size = filter_n * filter_c * filter_h * filter_w * sizeof(_Float16);
    size_t output_size = out_n * out_c * out_h * out_w * sizeof(_Float16);

    _Float16* h_input = (_Float16*)malloc(input_size);
    _Float16* h_filter = (_Float16*)malloc(filter_size);
    _Float16* h_output = (_Float16*)malloc(output_size);

    // 初始化数据
    memset(h_input, (_Float16)1, input_size);
    memset(h_filter, (_Float16)1, filter_size);
    memset(h_output, (_Float16)0, output_size);

    _Float16 *input, *filter, *output;
    hipMalloc((void **)&input, input_size);
    hipMalloc((void **)&filter, filter_size);
    hipMalloc((void **)&output, output_size);

    hipMemcpy(input, h_input, input_size, hipMemcpyHostToDevice);
    hipMemcpy(filter, h_filter, filter_size, hipMemcpyHostToDevice);
    hipMemcpy(output, h_output, output_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    // 计算所需工作空间大小
    size_t workspace_size = 0;
    MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(handle, input_desc, filter_desc,  conv_desc, output_desc, &workspace_size));
    void* workspace{nullptr};
    if (workspace_size > 0) {
        // std::cout << "Workspace size: " << (workspace_size / 1048576.0) << "MB"
        //       << std::endl;
        hipMalloc((void **)(&workspace), workspace_size);
    }

    // 搜索最优前向算法
    const int request_algo_count = 4;
    int returned_algo_count = -1;
    miopenConvAlgoPerf_t perf_results[4];
    MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(
        handle, input_desc, input, filter_desc, filter, conv_desc, output_desc, output,
        request_algo_count, &returned_algo_count, perf_results, workspace, workspace_size, true));

    // 打印最优算法信息
    // std::cout << "Found " << returned_algo_count << " algorithms." << std::endl;
    // for (int i = 0; i < returned_algo_count; ++i) {
    //     std::cout << "Algorithm " << i << ": "
    //               << "Time = " << perf_results[i].time << " ms, "
    //               << "Memory = " << (unsigned long long)perf_results[i].memory << " bytes" << std::endl;
    // }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    std::vector<float> elapsed_times(10);
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CALL(miopenConvolutionForward(handle, &alpha, input_desc, input, filter_desc, filter, conv_desc,
                                         perf_results[0].fwd_algo, &beta, output_desc, output, workspace, workspace_size));
    hipDeviceSynchronize();
    // 执行最优算法
    for (int i = 0; i < 10; ++i) {
        hipEventRecord(start, 0);
        MIOPEN_CALL(miopenConvolutionForward(handle, &alpha, input_desc, input, filter_desc, filter, conv_desc,
                                         perf_results[0].fwd_algo, &beta, output_desc, output, workspace, workspace_size));
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed_times[i], start, stop);
        // std::cout << "repeat " << i << " Convolution forward executed successfully!" << std::endl;
    }
    std::sort(elapsed_times.begin(), elapsed_times.end());
    // float median_time_ms = elapsed_times[elapsed_times.size() / 2];
    float median_time_ms = elapsed_times[0];
    // printf("%d, %d, %d, %d, %d, %d, %d\n", n, out_w, out_h, out_c, filter_w, filter_h, c);
	double flops = (double)n*(double)out_w*(double)out_h*(double)out_c*(double)filter_w*
            (double)filter_h*(double)c*2;
    double TFLOPS = flops / 1e12 / (median_time_ms / 1000);
    // long double FLOPS = 1 / 1e12 *n*out_h*out_w*filter_n*filter_h*filter_w*c*2;
    // long double tflops = FLOPS / (median_time_ms / 1000) ;
    std::cout << "Median elapsed time: " << median_time_ms << std::endl;
    // std::cout << "Median elapsed time: " << median_time_ms << " ms, Tflops: " << tflops << std::endl;
    printf("TFLOPS:%.4lf \n", TFLOPS);

    // 释放资源
    hipFree(input);
    hipFree(filter);
    hipFree(output);
    if (workspace) {
        hipFree(workspace);
    }
    MIOPEN_CALL(miopenDestroyTensorDescriptor(input_desc));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(filter_desc));
    MIOPEN_CALL(miopenDestroyTensorDescriptor(output_desc));
    MIOPEN_CALL(miopenDestroyConvolutionDescriptor(conv_desc));
    MIOPEN_CALL(miopenDestroy(handle));
}

int main(int argc, char* argv[]) {
	int caseidx = -1;
	if(argc == 2)
		caseidx = std::atoi(argv[1]);
	
	if(caseidx >=0 && caseidx <= 5){
		// printf("test case: %d, batch-m-k-n: %d, %d, %d, %d\n", caseidx, batch, m, k, n);
		printf("conv2d test case: %d\n", caseidx);
		RunConv2DExample(caseidx);		
	}
	else if(caseidx == -1){
		for(int i=0; i<6; i++){
			printf("---------------------------------------------------\n");
			printf("conv2d test case: %d\n", i);
			RunConv2DExample(i);
		}
	}
	else
		printf("invalid argument, case idx must between 0 and 5\n");
    return 0;
}
