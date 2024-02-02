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
	PrePostProcessor();
	~PrePostProcessor();

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
	/// <returns></returns>
	virtual std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor) = 0;
};

}
