#include "YOLOv8Postprocessor.cuh"
#include <algorithm>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>

#include "Matrix2D.hpp"


/*
static std::vector<DetectionBox> NonMaxSuppression(const std::vector<DetectionBox>& detections, float iouThreshold) {
    std::vector<DetectionBox> nmsDetections;
    std::vector<DetectionBox> dets = detections;
};


std::vector<DetectionBox> YOLOv8PostProcess(const float* d_rawData, int maxNumObj, int numClasses, float confThreshold, float iouThreshold)
{

}


struct ImageShape
{
    int width;
    int height;
};
*/

/*
    解析原始数据，并过滤低置信度的检测框
    d_rawdata：原始数据
    d_outdata：输出数据
    maxOutObj：最大输出检测框数量, 表示的是d_outdata输出数据输出内存中最大可以存储的检测框数量
    numChannels：通道数量
    numClasses：类别数量
    confThreshold：置信度阈值
*/
/*
__device__ void YOLOv8ParseRawDataKernel(const float* d_rawdata, float* d_outdata, int maxOutObj, int numChannels, int numClasses, float confThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numChannels || idx >= maxOutObj) return;

    Matrix2D<float, true> rawPtr(d_rawdata, numClasses + YOLOV8OutputElement, numChannels); // [1, 84, 8400]，  84 = 80 + 4

    float cx = rawPtr(idx, 0);
    float cy = rawPtr(idx, 1);
    float w = rawPtr(idx, 2);
    float h = rawPtr(idx, 3);

    int classIdx = -1;
    float maxClassScore = 0.f;

    for (int j = YOLOV8OutputElement; j < numClasses + YOLOV8OutputElement; ++j) {
        float score = rawPtr(idx, j);
        if (score > confThreshold && score > maxClassScore) {
            maxClassScore = score;
            classIdx = j - YOLOV8OutputElement;
        }
    }

    if (classIdx == -1) return;
    atomicAdd(d_outdata, 1);

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;


    d_outdata[idx * 4 + 0] = x1;
    d_outdata[idx * 4 + 1] = y1;
    d_outdata[idx * 4 + 2] = x2;
    d_outdata[idx * 4 + 3] = y2;

    d_outdata[idx * 4 + 4] = classIdx;
    d_outdata[idx * 4 + 5] = maxClassScore;
}


__global__ void YOLOv8ParseRawData(const float* d_rawdata, float* d_outdata, int maxOutObj, int numChannels, int numClasses, float confThreshold)
{
    int numThreads = 1024;
    int numBlocks = (numChannels + numThreads - 1) / numThreads;

    YOLOv8ParseRawDataKernel << <numBlocks, numThreads >> > (d_rawdata, d_outdata, maxOutObj, numChannels, numClasses, confThreshold);
}



__device__ void NMSBatchKernel(const float* d_rawdata, float iouThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxObj = (int)d_rawdata[0]; // 解析出的最大的检测框数量
    if (idx >= maxObj) return;


    auto cur = reinterpret_cast<const DetectionBox*>(&d_rawdata[1 + idx * RAW_YOLOV8_OUTPUTS_ELEMENT]);
    if (cur->suppressed) return;



}


__device__ void RestoreCoordinateKernel(DetectionBox* d_boxes, int numBoxes,
    int origWidth, int origHeight, int inputWidth, int inputHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;

    DetectionBox& box = d_boxes[idx];

    // 计算缩放比例和填充偏移
    float scale = __min(static_cast<float>(inputWidth) / origWidth, static_cast<float>(inputHeight) / origHeight);
    float pad_x = (inputWidth - origWidth * scale) / 2.0f;
    float pad_y = (inputHeight - origHeight * scale) / 2.0f;

    // 去除填充并还原比例
    box.x1 = (box.x1 - pad_x) / scale;
    box.y1 = (box.y1 - pad_y) / scale;
    box.x2 = (box.x2 - pad_x) / scale;
    box.y2 = (box.y2 - pad_y) / scale;

    // 裁剪到图像边界
    box.x1 = fmaxf(0.0f, fminf(static_cast<float>(origWidth), box.x1));
    box.y1 = fmaxf(0.0f, fminf(static_cast<float>(origHeight), box.y1));
    box.x2 = fmaxf(0.0f, fminf(static_cast<float>(origWidth), box.x2));
    box.y2 = fmaxf(0.0f, fminf(static_cast<float>(origHeight), box.y2));
}


__global__ void RestoreCoordinate(DetectionBox* d_boxes, int numBoxes,
    int origWidth, int origHeight, int inputWidth, int inputHeight) {
    int numThreads = 1024;
    int numBlocks = (numBoxes + numThreads - 1) / numThreads;

    RestoreCoordinateKernel << <numBlocks, numThreads >> > (d_boxes, numBoxes, origWidth, origHeight, inputWidth, inputHeight);
}

*/