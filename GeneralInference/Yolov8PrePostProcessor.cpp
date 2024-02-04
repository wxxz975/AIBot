#include "Yolov8PrePostProcessor.h"


#include "Utils.h"

namespace GeneralInference {

	Yolov8PrePostProcessor::Yolov8PrePostProcessor(std::shared_ptr<Model>& model_io_info)
		:PrePostProcessor(model_io_info)
	{
		m_ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	}

	Yolov8PrePostProcessor::~Yolov8PrePostProcessor()
	{
	}

	std::vector<Ort::Value> Yolov8PrePostProcessor::Preprocess(const cv::Mat& image)
	{
		std::vector<Ort::Value> result;
		m_original_shape = image.size();
		const std::vector<int64_t>& input_tensor_shape = GetInputTensorShape();
		cv::Mat rgb_image, resized_image, float_image;


		rgb_image = Convert2RGB(image);

		resized_image = Letterbox(rgb_image, GetCurrentShape());
		
		resized_image.convertTo(float_image, CV_32FC3, 1 / 255.0);

		HWC2CHW(float_image, m_blob);

		result.push_back(
			Ort::Value::CreateTensor<float>(m_ort_memory_info,
				m_blob.data(), m_blob.size(),
				input_tensor_shape.data(), input_tensor_shape.size())
		);

		return result;
	}


	std::vector<BoundingBox> Yolov8PrePostProcessor::Postprocess(const std::vector<Ort::Value>& output_tensor, float conf_threshold, float iou_threshold)
	{
		std::vector<BoundingBox> boxes = ParseRawData(output_tensor, conf_threshold);
		std::vector<BoundingBox> nms_boxes = NMS(boxes, iou_threshold);

		RestoreOriginalCoordsInBatch(GetCurrentShape(), GetOriginalShape(), nms_boxes);

		return nms_boxes;
	}


	std::vector<BoundingBox> Yolov8PrePostProcessor::ParseRawData(const std::vector<Ort::Value>& output_tensor, float conf_threshold)
	{
		

	}
	cv::Size Yolov8PrePostProcessor::GetCurrentShape() const
	{
		std::vector<int64_t> shape = m_model_io_info->input.shapes.at(0);
		return {shape.at(2), shape.at(3)};
	}
	cv::Size Yolov8PrePostProcessor::GetOriginalShape() const
	{
		return m_original_shape;
	}
	std::vector<int64_t> Yolov8PrePostProcessor::GetInputTensorShape() const
	{
		return m_model_io_info->input.shapes.at(0);
	}
}