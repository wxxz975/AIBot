#include "Yolov5PrePostProcessor.h"



namespace GeneralInference{

	Yolov5PrePostProcessor::Yolov5PrePostProcessor()
	{
	}

	Yolov5PrePostProcessor::~Yolov5PrePostProcessor()
	{
	}

	std::vector<Ort::Value> Yolov5PrePostProcessor::Preprocess(const cv::Mat& image)
	{
		return std::vector<Ort::Value>();
	}


	std::vector<BoundingBox> Yolov5PrePostProcessor::Postprocess(const std::vector<Ort::Value>& out_tensor)
	{
		return std::vector<BoundingBox>();
	}
}