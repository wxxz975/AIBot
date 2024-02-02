#pragma once
#include <string>

#include "PrePostProcessor.h"

namespace GeneralInference
{
	class Yolov5PrePostProcessor: public PrePostProcessor
	{
	public:
		explicit Yolov5PrePostProcessor();
		~Yolov5PrePostProcessor();


		std::vector<Ort::Value> Preprocess(const cv::Mat& image) override;

		std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor) override;
	private:

	};

	
}