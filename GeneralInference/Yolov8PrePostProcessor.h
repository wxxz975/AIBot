#pragma once

#include <string>

#include "PrePostProcessor.h"

namespace GeneralInference
{
	class Yolov8PrePostProcessor : public PrePostProcessor
	{
	public:
		explicit Yolov8PrePostProcessor(std::shared_ptr<Model>& model_io_info);
		~Yolov8PrePostProcessor();


		std::vector<Ort::Value> Preprocess(const cv::Mat& image) override;

		std::vector<BoundingBox> Postprocess(const std::vector<Ort::Value>& out_tensor, float conf_threshold, float iou_threshold) override;
		
	private:
		std::vector<BoundingBox> ParseRawData(const std::vector<Ort::Value>& output_tensor, float conf_threshold);

		/// <summary>
		/// The image shape after letterbox is the input image shape required by the neural network.
		/// </summary>
		/// <returns></returns>
		cv::Size GetCurrentShape() const;
		
		/// <summary>
		/// Original image shape without any processing
		/// </summary>
		/// <returns></returns>
		cv::Size GetOriginalShape() const;


		std::vector<int64_t> GetInputTensorShape() const;

	private:
		Ort::MemoryInfo m_ort_memory_info{ nullptr };
		cv::Size m_original_shape{};

		std::vector<float> m_blob;
	};

}