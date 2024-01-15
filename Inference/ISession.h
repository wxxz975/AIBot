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
    /// @brief ��ʼ�� ����Ự �� ����ģ��
    /// @param modelPath ģ��·��
    /// @return �����Ƿ��ʼ���ɹ�
    virtual bool Initialize(const std::string& modelPath) = 0;

    /// @brief ģ���������
    /// @param image �����ͼ��
    /// @return ����������ɵĽ��
    virtual std::vector<ResultNode> Infer(const cv::Mat& image) = 0;

    /// @brief ��ȡģ�Ͳ���
    /// @return 
    virtual Model* GetModel() { return model_; };

    /// @brief ����confidence ��ֵ
    /// @param conf ���Ŷ�
    void SetConfidence(float conf) { confidenceThreshold_ = conf; };

    /// @brief ����iou��ֵ
    /// @param iou ��������ֵ, ���ڷǼ������Ƶ���ֵ����
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