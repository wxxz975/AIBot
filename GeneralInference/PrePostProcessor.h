#pragma once

#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "GeneralStruct.h"



namespace GeneralInference
{ 


class PrePostProcessor
{
public:
	PrePostProcessor(std::shared_ptr<Model>& model_io_info) : m_model_io_info(model_io_info) {};
	virtual ~PrePostProcessor() {};

public:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	virtual std::vector<Ort::Value> Preprocess(const cv::Mat& image) = 0;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="out_tensor"></param>
	/// <param name="conf_threshold"></param>
	/// <param name="iou_threshold"></param>
	/// <returns></returns>
	virtual std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor, float conf_threshold = 0.5, float iou_threshold = 0.45) = 0;
	
protected:
	std::shared_ptr<Model> m_model_io_info;
};

}
