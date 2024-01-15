#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "YoloDefine.h"


/// @brief 绘制框和标签结果到图像上
/// @param image 需要绘制的图像
/// @param boxes 推理返回结果
/// @param labels 所有标签列表
/// @return 返回绘制好的图像
inline cv::Mat RenderBoundingBoxes(const cv::Mat& image, const std::vector<ResultNode>& boxes,
    const std::vector<std::string>& labels = {})
{
    cv::Mat out = image.clone();
    for (const auto& box : boxes) {
        cv::rectangle(out, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2); // 绘制绿色边界框，线宽为2

        cv::Point labelPosition(box.x, box.y - 10); // 调整标签位置，使其位于边界框上方
        cv::putText(out, labels.empty() ?  std::to_string(box.classIdx) : labels[box.classIdx], labelPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    return out;
}