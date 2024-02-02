#include "Yolov8PrePostProcessor.h"




namespace GeneralInference {

	Yolov8PrePostProcessor::Yolov8PrePostProcessor()
	{
	}

	Yolov8PrePostProcessor::~Yolov8PrePostProcessor()
	{
	}

	std::vector<Ort::Value> Yolov8PrePostProcessor::Preprocess(const cv::Mat& image)
	{
		return std::vector<Ort::Value>();
	}


	std::vector<BoundingBox> Yolov8PrePostProcessor::Postprocess(const std::vector<Ort::Value>& out_tensor)
	{
		return std::vector<BoundingBox>();
	}
}