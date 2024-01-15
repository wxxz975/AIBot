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


    /// @brief  yolov5 ģ�͵�Ԥ������(��һ��ͼ��洢��ʽΪRGB��ͼ���СΪģ��ָ����С������Onnxruntime��Ҫ��Ort::Value���Ͳ��ҷ���)
   /// @param image ��Ҫ�����Ԥ����ͼ��
   /// @return ���ؿ�ֱ�������Ort��ֵ
    std::vector<Ort::Value> Preprocess(const cv::Mat& image);

    /// @brief yolov5����(��Ҫ�Ƕ�ȡԭʼonnxruntime���ɵ����ݲ�������nms���� �ĵ�������ֵ�Ľ�����ϲ�����)
    /// @param outTensor ����������tensor
    /// @param originalImageShape ԭʼͼ���shape
    /// @param confThreshold ���Ŷ���ֵ
    /// @param iouThreshold iou��ֵ
    /// @return ���ش������������ݣ���������x,y,w,h, ���index, ���Ŷ�
    std::vector<ResultNode> Postprocess(const std::vector<Ort::Value>& outTensor,
        const cv::Size& originalImageShape,
        float confThreshold, float iouThreshold);

    /// @brief ��ͼ���һ��Ϊͳһ��С����Ҫ�Ƿ������ģ�͵�����ά�ȵĳߴ�
    /// @param image �����ͼ��
    /// @param newShape ��Ҫת�������µ�shape
    /// @param color ������ɫ
    /// @param scaleFill �Ƿ���Ҫ���
    /// @param scaleUp 
    /// @param stride ����
    /// @return ���ش�����ͼ��
    cv::Mat Letterbox(const cv::Mat& image, const cv::Size& newShape = cv::Size(640, 640),
        const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scaleFill = false, bool scaleUp = true,
        int stride = 32);


    /// @brief ͼ����opencv��ȡ�ĸ�ʽ�� ͳһת��ΪRGB
    /// @param image ��Ҫת����ͼ��
    /// @param outImage ��Ҫ�������ͼ��
    /// @return �����Ƿ�ת���ɹ�
    bool ConvertToRGB(const cv::Mat& image, cv::Mat& outImage);


    /// @brief ����ԭʼͼ���е�����
    /// @param currentShape 
    /// @param originalShape 
    /// @param outCoords 
    void GetOriCoords(const cv::Size& currentShape,
        const cv::Size& originalShape, cv::Rect& outCoords);

    /// @brief ��ȡ ��ǰ���� score ��ߵ�һ��
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

