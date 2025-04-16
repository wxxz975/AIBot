#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "YOLOv8Definitions.cuh"



/*
    1、解析原始数据并且过滤低置信度的检测框
    2、使用NMS过滤冗余的检测框
    3、将检测框还原到原始图像上
*/



/*

std::vector<DetectionBox> PostProcessDetections(
    const float* rawData,
    int numDetections,
    int numClasses,
    float confThreshold,
    float iouThreshold,
    float scale);*/