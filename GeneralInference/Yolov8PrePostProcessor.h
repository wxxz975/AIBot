#pragma once

#include <string>

#include "PrePostProcessor.h"

namespace GeneralInference
{
	class Yolov8PrePostProcessor : public PrePostProcessor
	{
	public:
		explicit Yolov8PrePostProcessor();
		~Yolov8PrePostProcessor();


		std::vector<Ort::Value> Preprocess(const cv::Mat& image) override;

		std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor) override;

	private:

	};

}