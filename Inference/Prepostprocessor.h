#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "YoloDefine.h"
#include "Model.h"

class Prepostprocessor
{
public:
	Prepostprocessor(Model* model);
	~Prepostprocessor();


    /// @brief  yolov5 模型的预处理函数(归一化图像存储格式为RGB、图像大小为模型指定大小、生成Onnxruntime需要的Ort::Value类型并且返回)
   /// @param image 需要输入的预处理图像
   /// @return 返回可直接输入给Ort的值
    std::vector<Ort::Value> Preprocess(const cv::Mat& image);

    /// @brief yolov5后处理(主要是读取原始onnxruntime生成的数据并解析后经nms处理 的到符合阈值的结果集合并返回)
    /// @param outTensor 推理后输出的tensor
    /// @param originalImageShape 原始图像的shape
    /// @param confThreshold 置信度阈值
    /// @param iouThreshold iou阈值
    /// @return 返回处理后的最终数据，包含坐标x,y,w,h, 类别index, 置信度
    std::vector<ResultNode> Postprocess(const std::vector<Ort::Value>& outTensor,
        const cv::Size& originalImageShape,
        float confThreshold, float iouThreshold);

    /// @brief 将图像归一画为统一大小，主要是符合这个模型的输入维度的尺寸
    /// @param image 输入的图像
    /// @param newShape 需要转换到的新的shape
    /// @param color 填充的颜色
    /// @param scaleFill 是否需要填充
    /// @param scaleUp 
    /// @param stride 步长
    /// @return 返回处理后的图像
    cv::Mat Letterbox(const cv::Mat& image, const cv::Size& newShape = cv::Size(640, 640),
        const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scaleFill = false, bool scaleUp = true,
        int stride = 32);


    /// @brief 图像由opencv读取的格式， 统一转换为RGB
    /// @param image 需要转换的图像
    /// @param outImage 需要输出到的图像
    /// @return 返回是否转换成功
    bool ConvertToRGB(const cv::Mat& image, cv::Mat& outImage);


    /// @brief 计算原始图像中的坐标
    /// @param currentShape 
    /// @param originalShape 
    /// @param outCoords 
    void GetOriCoords(const cv::Size& currentShape,
        const cv::Size& originalShape, cv::Rect& outCoords);

    /// @brief 获取 当前框中 score 最高的一个
    /// @param it 
    /// @param numClasses 
    /// @param bestConf 
    /// @param bestClassId 
    void GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);

    /// @brief 
    /// @param tensor 
    /// @param conf_threshold 
    /// @param boxes 
    /// @param confs 
    /// @param classIds 
    void ParseRawOutput(const std::vector<Ort::Value>& tensor, float conf_threshold, std::vector<cv::Rect>& boxes, std::vector<float>& confs, std::vector<int>& classIds);

private:

    Model* model_ = nullptr;

    float* blob_ = nullptr;
    size_t blobSize_ = 0;

    Ort::MemoryInfo memInfo_{ nullptr };
};

