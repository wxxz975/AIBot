#pragma once
#include <string>

#include "PrePostProcessor.h"

namespace GeneralInference
{
	class Yolov5PrePostProcessor: public PrePostProcessor
	{
	public:
		explicit Yolov5PrePostProcessor(std::shared_ptr<Model>& model_io_info);
		~Yolov5PrePostProcessor();


		std::vector<Ort::Value> Preprocess(const cv::Mat& image) override;

		std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor, float conf_threshold, float iou_threshold) override;
	
	private:

		/// <summary>
		/// Parse the original data and filter out boxes with low confidence, 
		/// and finally find the optimal category in the box
		/// </summary>
		/// <param name="output_tensor"></param>
		/// <param name="conf_threshlod"></param>
		/// <returns></returns>
		std::vector<BoundingBox> ParseRawData(const std::vector<Ort::Value>& output_tensor, float conf_threshlod);
		

		cv::Size GetCurrentShape() const;
		cv::Size GetOriginalShape() const;

		std::vector<int64_t> GetInputTensorShape()const;

	private:

		cv::Size m_original_shape;
		cv::Size m_current_shape;
		std::vector<float> m_blob;

		Ort::MemoryInfo m_ort_memory_info{nullptr};
	};

	
}