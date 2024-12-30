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
size_t pictureDataSize = 0;
void *pictureHostData;
void *pictureDeviceData;
aclmdlDataset *inputDataSet;
aclDataBuffer *inputDataBuffer;
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
void ReadPictureTotHost(const char *picturePath)
{
	string fileName = picturePath;
	ifstream binFile(fileName, ifstream::binary);
	if(!binFile)
		printf("load file error!\n");
	binFile.seekg(0, binFile.end);
	pictureDataSize = binFile.tellg();
	binFile.seekg(0, binFile.beg);
	aclError ret = aclrtMallocHost(&pictureHostData, pictureDataSize);
	binFile.read((char *)pictureHostData, pictureDataSize);
	binFile.close();
}

// 申请Device侧的内存，再以内存复制的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice()
{
	aclError ret = aclrtMalloc(&pictureDeviceData, pictureDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(pictureDeviceData, pictureDataSize, pictureHostData, pictureDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

void LoadPicture(const char *picturePath)
{
	ReadPictureTotHost(picturePath);
	CopyDataFromHostToDevice();
}

// 准备模型推理的输入数据结构
void CreateModelInput()
{
	// 创建aclmdlDataset类型的数据，描述模型推理的输入
	inputDataSet = aclmdlCreateDataset();
	inputDataBuffer = aclCreateDataBuffer(pictureDeviceData, pictureDataSize);
	aclError ret = aclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer);
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
		1, 96, 96, 55, 27, 3, 2, 0, 1,
		1, 256, 256, 27, 13, 3, 2, 0, 1,
		1, 256, 256, 13, 6, 3, 2, 0, 1,
		1, 64, 64, 112, 56, 3, 2, 0, 1,
		1, 2048, 2048, 7, 1, 7, 1, 0, 1,
		1, 256, 256, 20, 20, 5, 1, 2, 1
	};

	int64_t* args = args_v.data() + caseidx*9;

	int64_t batch = args[0], inC = args[1], outC = args[2], inHnW = args[3], outHnW = args[4], 
		kernel_size = args[5], stride = args[6], padding = args[7], dilation = args[8];

  double tflops = (int64_t)batch*(int64_t)outHnW*(int64_t)outHnW*(int64_t)outC;
  tflops = tflops * (int64_t)kernel_size*(int64_t)kernel_size;
  tflops = tflops / 1e12;
  tflops = tflops / midt;

	printf("case:%d, inputshape: (%ld, %ld, %ld, %ld)\n", caseidx, batch, inC, inHnW, inHnW);
  printf("time:%.6f, TFLOPS:%.4f\n", midt, tflops);
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
	aclError ret = aclrtFreeHost(pictureHostData);
	pictureHostData = nullptr;
	ret = aclrtFree(pictureDeviceData);
	pictureDeviceData = nullptr;
	aclDestroyDataBuffer(inputDataBuffer);
	inputDataBuffer = nullptr;
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

	const char *modelPath = "../model/pool0.om";
	LoadModel(modelPath);

	std::string picturePath = "../data/input" + std::to_string(caseidx) + ".bin";
	std::cout << picturePath << "\n";
	LoadPicture(picturePath.c_str());

	Inference(caseidx);

	UnloadModel();
	UnloadPicture();
	DestroyResource();
}
