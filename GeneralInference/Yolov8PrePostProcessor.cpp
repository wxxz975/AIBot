#include "Yolov8PrePostProcessor.h"


#include "Utils.h"
#include <opencv2/dnn.hpp>


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
		m_blob = Letterbox(image, GetCurrentShape());
		m_blob = cv::dnn::blobFromImage(m_blob, 1 / 255.0, GetCurrentShape(), cv::Scalar(0,0,0), true);

		result.push_back(
			Ort::Value::CreateTensor<float>(m_ort_memory_info,
				m_blob.ptr<float>(),
				m_blob.total(),
				input_tensor_shape.data(), input_tensor_shape.size()
			)
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
 		std::vector<BoundingBox> result;
		// output only one Dimensions
		const float* raw_ptr = output_tensor.at(0).GetTensorData<float>();
		//std::size_t count = output_tensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<int64_t> output_shape = output_tensor.at(0).GetTensorTypeAndShapeInfo().GetShape();
		
		std::size_t num_channels = output_shape.at(1);	// cols
		std::size_t num_anchors = output_shape.at(2);	// rows
		std::size_t num_classes = num_channels - YOLOV8_OUTBOX_ELEMENT_COUNT;

		cv::Size newShape = cv::Size(num_anchors, num_channels);
	
		cv::Mat output0 = cv::Mat(newShape, CV_32F, (float*)raw_ptr).t();

		float* pData = output0.ptr<float>();

		for (std::size_t idx = 0; idx < output0.rows; ++idx)
		{
			cv::Mat scores = output0.row(idx).colRange(YOLOV8_OUTBOX_ELEMENT_COUNT, num_channels);
			const Yolov8RawResult* bbox = output0.row(idx).ptr<Yolov8RawResult>();
			cv::Point classIdPoint;
			double score;
			cv::minMaxLoc(scores, 0, &score, 0, &classIdPoint);
			if (score > conf_threshold)
			{
				std::int32_t width = static_cast<std::int32_t>(bbox->w);
				std::int32_t height = static_cast<std::int32_t>(bbox->h);
				std::int32_t left = static_cast<std::int32_t>(bbox->cx) - width / 2;
				std::int32_t top = static_cast<std::int32_t>(bbox->cy) - height / 2;

				result.push_back({ left, top, width, height,
					static_cast<std::size_t>(classIdPoint.x),
					static_cast<float>(score) });
			}
		}
		return result;
	}


	cv::Size Yolov8PrePostProcessor::GetCurrentShape() const
	{
		std::vector<int64_t> shape = m_model_io_info->input.shapes.at(0);
		return { static_cast<int>(shape.at(2)), static_cast<int>(shape.at(3))};
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