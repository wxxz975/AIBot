#pragma once

#include "General.h"
#include <opencv2/opencv.hpp>


namespace InferenceEngine
{
    class PrePostProcessor
    {
    private:
        /* data */
    public:
        PrePostProcessor() = default;
        virtual ~PrePostProcessor() = default;

        /// @brief 对图像进行预处理，返回符合模型输入的数据
        /// @param image 
        /// @return 
        virtual ImageDataPtr Preprocessing(const cv::Mat& image) = 0;

        /// @brief 对推理引擎输出的结果进行解析
        /// @param output 
        /// @return 
        virtual std::vector<BoundingBox> Postprocessing(RawOutputPtrVec output) = 0;
    };

};

