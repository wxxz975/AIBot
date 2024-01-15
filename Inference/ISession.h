#pragma once
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "YoloDefine.h"
#include "Model.h"

class ISession
{
public:
    ISession() = default;
    ~ISession() = default;


public:
    /// @brief 初始化 推理会话 、 解析模型
    /// @param modelPath 模型路径
    /// @return 返回是否初始化成功
    virtual bool Initialize(const std::string& modelPath) = 0;

    /// @brief 模型推理入口
    /// @param image 输入的图像
    /// @return 返回推理完成的结果
    virtual std::vector<ResultNode> Infer(const cv::Mat& image) = 0;

    /// @brief 获取模型参数
    /// @return 
    virtual Model* GetModel() { return model_; };

    /// @brief 设置confidence 阈值
    /// @param conf 置信度
    void SetConfidence(float conf) { confidenceThreshold_ = conf; };

    /// @brief 设置iou阈值
    /// @param iou 交并比阈值, 用于非极大抑制的阈值控制
    void SetIOU(float iou) { iouThreshold_ = iou; };


protected:
    virtual bool WarmUpModel() = 0;

protected:
    Model* model_;

    float confidenceThreshold_ = 0.5;
    float iouThreshold_ = 0.45;

    Ort::Session session_{ nullptr };
    Ort::SessionOptions sessionOpt{ nullptr };
    Ort::Env env_{ nullptr };
    std::string envName_;
};