#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;
int32_t deviceId = 0;
uint32_t modelId;
size_t ADataSize = 0;
size_t BDataSize = 0;
void *AHostData;
void *BHostData;
void *ADeviceData;
void *BDeviceData;
aclmdlDataset *inputDataSet;
aclDataBuffer *ADataBuffer;
aclDataBuffer *BDataBuffer;
aclmdlDataset *outputDataSet;
aclDataBuffer *outputDataBuffer;
aclmdlDesc *modelDesc;
size_t outputDataSize = 0;
void *outputDeviceData;
void *outputHostData;

void InitResource()
{
	aclError ret = aclInit(nullptr);
	ret = aclrtSetDevice(deviceId);
}

void LoadModel(const char *modelPath)
{
	aclError ret = aclmdlLoadFromFile(modelPath, &modelId);
}

// 申请内存，使用C/C++标准库的函数将测试图片读入内存
void ReadPictureTotHost(const char *picturePath1, const char *picturePath2)
{
	{
		string fileName = picturePath1;
		ifstream binFile(fileName, ifstream::binary);
		if(!binFile)
			printf("load file error!\n");
		binFile.seekg(0, binFile.end);
		ADataSize = binFile.tellg();
		binFile.seekg(0, binFile.beg);
		aclError ret = aclrtMallocHost(&AHostData, ADataSize);
		binFile.read((char *)AHostData, ADataSize);
		binFile.close();		
	}

	{
		string fileName2 = picturePath2;
		ifstream binFile2(fileName2, ifstream::binary);
		if(!binFile2)
			printf("load file error!\n");
		binFile2.seekg(0, binFile2.end);
		BDataSize = binFile2.tellg();
		binFile2.seekg(0, binFile2.beg);
		aclError ret = aclrtMallocHost(&BHostData, BDataSize);
		binFile2.read((char *)BHostData, BDataSize);
		binFile2.close();		
	}

}

// 申请Device侧的内存，再以内存复制的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice()
{
	aclError ret = aclrtMalloc(&ADeviceData, ADataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(ADeviceData, ADataSize, AHostData, ADataSize, ACL_MEMCPY_HOST_TO_DEVICE);

	ret = aclrtMalloc(&BDeviceData, BDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(BDeviceData, BDataSize, BHostData, BDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

void LoadPicture(const char *picturePath1, const char *picturePath2)
{
	ReadPictureTotHost(picturePath1, picturePath2);
	CopyDataFromHostToDevice();
}

// 准备模型推理的输入数据结构
void CreateModelInput()
{
	// 创建aclmdlDataset类型的数据，描述模型推理的输入
	inputDataSet = aclmdlCreateDataset();
	ADataBuffer = aclCreateDataBuffer(ADeviceData, ADataSize);
	BDataBuffer = aclCreateDataBuffer(BDeviceData, BDataSize);
	aclError ret = aclmdlAddDatasetBuffer(inputDataSet, ADataBuffer);
	ret = aclmdlAddDatasetBuffer(inputDataSet, BDataBuffer);
}

// 准备模型推理的输出数据结构
void CreateModelOutput()
{
	// 创建模型描述信息
	modelDesc = aclmdlCreateDesc();
	aclError ret = aclmdlGetDesc(modelDesc, modelId);
	// 创建aclmdlDataset类型的数据，描述模型推理的输出
	outputDataSet = aclmdlCreateDataset();
	// 获取模型输出数据需占用的内存大小，单位为Byte
	outputDataSize = aclmdlGetOutputSizeByIndex(modelDesc, 0);
	// 申请输出内存
	ret = aclrtMalloc(&outputDeviceData, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	outputDataBuffer = aclCreateDataBuffer(outputDeviceData, outputDataSize);
	ret = aclmdlAddDatasetBuffer(outputDataSet, outputDataBuffer);
}

// 执行模型
void Inference(int caseidx)
{
	CreateModelInput();
	CreateModelOutput();

  std::vector<float> result;
  for(int i=0; i<20; i++){
    auto start = std::chrono::high_resolution_clock::now();

		aclError ret = aclmdlExecute(modelId, inputDataSet, outputDataSet);

		aclrtSynchronizeDevice();
		if (ret != ACL_SUCCESS)
			printf("error! code:%d\n", ret);
		
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    float elapsedTime = duration.count();

    result.push_back(elapsedTime);
  }
  std::sort(result.begin(), result.end());
  double midt = result[(result.size()-1) / 2];
	
	vector<int64_t> args_v = {
			1, 1, 4096, 4096,
			16, 20, 32, 20,
			2, 20, 256, 768,
			2, 20, 256, 1024,
			128, 1, 128, 58,
			128, 1, 58, 128,
		};
	
	int64_t* args = args_v.data() + caseidx*4;

	int64_t batch = args[0], M = args[1], K = args[2], N = args[3];

  double tflops = (int64_t)batch*(int64_t)M*(int64_t)K*(int64_t)N*2;
  tflops = tflops / midt;
  tflops = tflops / 1e12;

	printf("case:%d, args: (%ld, %ld, %ld, %ld)\n", caseidx, batch, M, K, N);
  printf("time:%.6f ms, tflops:%.3f\n", midt*1000, tflops);
}

void UnloadModel()
{
	// 释放模型描述信息
	aclmdlDestroyDesc(modelDesc);
	// 卸载模型
	aclmdlUnload(modelId);
}

void UnloadPicture()
{
	aclError ret = aclrtFreeHost(AHostData);
	ret = aclrtFreeHost(BHostData);
	AHostData = nullptr;
	BHostData = nullptr;
	ret = aclrtFree(ADeviceData);
	ret = aclrtFree(BDeviceData);
	ADeviceData = nullptr;
	ADeviceData = nullptr;
	aclDestroyDataBuffer(ADataBuffer);
	aclDestroyDataBuffer(BDataBuffer);
	ADataBuffer = nullptr;
	BDataBuffer = nullptr;
	aclmdlDestroyDataset(inputDataSet);
	inputDataSet = nullptr;

	ret = aclrtFreeHost(outputHostData);
	outputHostData = nullptr;
	ret = aclrtFree(outputDeviceData);
	outputDeviceData = nullptr;
	aclDestroyDataBuffer(outputDataBuffer);
	outputDataBuffer = nullptr;
	aclmdlDestroyDataset(outputDataSet);
	outputDataSet = nullptr;
}

void DestroyResource()
{
	aclError ret = aclrtResetDevice(deviceId);
	aclFinalize();
}

int main(int argc, char* argv[])
{
	int caseidx = 0;
	if (argc = 2) {
		caseidx = std::stoi(argv[1]);
		printf("test case:%d\n", caseidx);
  }
	InitResource();

	const char *modelPath = "../model/matmul0.om";
	LoadModel(modelPath);

	std::string picturePath1 = "../data/A" + std::to_string(caseidx) + ".bin";
	std::string picturePath2 = "../data/B" + std::to_string(caseidx) + ".bin";
	// std::cout << picturePath1 << "; " << picturePath2 << "\n";
	LoadPicture(picturePath1.c_str(), picturePath2.c_str());

	Inference(caseidx);

	UnloadModel();
	UnloadPicture();
	DestroyResource();
}
