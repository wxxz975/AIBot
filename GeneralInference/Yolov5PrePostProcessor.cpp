#include "Yolov5PrePostProcessor.h"
#include "Utils.h"


namespace GeneralInference{

	Yolov5PrePostProcessor::Yolov5PrePostProcessor(std::shared_ptr<Model>& model_io_info)
		:PrePostProcessor(model_io_info)
	{
		m_ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	}

	Yolov5PrePostProcessor::~Yolov5PrePostProcessor()
	{
	
	}

	std::vector<Ort::Value> Yolov5PrePostProcessor::Preprocess(const cv::Mat& image)
	{
		std::vector<Ort::Value> result;
		cv::Mat rgb_image, letterbox_image, float_image;
		cv::Size floatImageSize;
		std::vector<int64_t> input_tensor_shape = GetInputTensorShape();
		m_original_shape = image.size();

		m_blob = Letterbox(image, GetCurrentShape());
		m_blob = cv::dnn::blobFromImage(m_blob, 1 / 255.0, GetCurrentShape(), cv::Scalar(0, 0, 0), true);
		/*
		// Convert image to rgb format
		rgb_image = Convert2RGB(image);

		// Convert the image to an image of a specified size using letterbox
		letterbox_image = Letterbox(rgb_image, GetCurrentShape());
		
		// Convert uint8_t type to float32 type and normalize to between 0-1
		letterbox_image.convertTo(float_image, CV_32FC3, 1 / 255.0);

		// Convert image color channel from hwc to chw
		HWC2CHW(float_image, m_blob);
		*/
		result.push_back(
			Ort::Value::CreateTensor<float>(m_ort_memory_info,
				//m_blob.data(), m_blob.size(),
				m_blob.ptr<float>(),
				m_blob.total(),
				input_tensor_shape.data(), input_tensor_shape.size())
		);

		return result;
	}


	std::vector<BoundingBox> Yolov5PrePostProcessor::Postprocess(const std::vector<Ort::Value>& output_tensor, float conf_threshold, float iou_threshold)
	{
		std::vector<BoundingBox> boxes = ParseRawData(output_tensor, conf_threshold);
		std::vector<BoundingBox> nms_boxes = NMS(boxes, iou_threshold);
		
		RestoreOriginalCoordsInBatch(GetCurrentShape(), GetOriginalShape(), nms_boxes);
		
		return nms_boxes;
	}


	std::vector<BoundingBox> Yolov5PrePostProcessor::ParseRawData(
		const std::vector<Ort::Value>& output_tensor, float conf_threshold)
	{
		std::vector<BoundingBox> result;
		// output only one Dimensions
		const float* raw_ptr = output_tensor.at(0).GetTensorData<float>();
		std::vector<int64_t> output_shape = output_tensor.at(0).GetTensorTypeAndShapeInfo().GetShape();
			
		// These all depend on the model, [1,25200,19]
		std::size_t num_channels = output_shape.at(2);
		std::size_t num_anchors = output_shape.at(1);
		cv::Size outShape = cv::Size(num_channels, num_anchors);
		cv::Mat output0 = cv::Mat(outShape, CV_32F, (float*)raw_ptr);

		for (std::size_t idx = 0; idx < output0.rows; ++idx)
		{
			cv::Mat scores = output0.row(idx).colRange(YOLOV5_OUTBOX_ELEMENT_COUNT, num_channels);
			const Yolov5RawResult* bbox = output0.row(idx).ptr<Yolov5RawResult>();
			cv::Point classIdPoint;
			double score;
			cv::minMaxLoc(scores, 0, &score, 0, &classIdPoint);

			if (bbox->box_conf > conf_threshold)
			{
				std::int32_t width = static_cast<std::int32_t>(bbox->w);
				std::int32_t height = static_cast<std::int32_t>(bbox->h);
				std::int32_t left = static_cast<std::int32_t>(bbox->cx) - width / 2;
				std::int32_t top = static_cast<std::int32_t>(bbox->cy) - height / 2;

				result.push_back({ left, top, width, height,
					static_cast<std::size_t>(classIdPoint.x),
					static_cast<float>(score * bbox->box_conf) });
			}
		}
		/*
		for (auto iter = output.begin(); iter != output.begin() + count; iter += num_channels)
		{
			const Yolov5RawResult* bbox = reinterpret_cast<const Yolov5RawResult*>(&(*iter));
			float boxConf = bbox->box_conf; // box confidence
			auto start_class_conf = iter + YOLOV5_OUTBOX_ELEMENT_COUNT;

			if (boxConf < conf_threshold)
				continue;

			std::int32_t width = static_cast<std::int32_t>(bbox->w);
			std::int32_t height = static_cast<std::int32_t>(bbox->h);
			std::int32_t left = static_cast<std::int32_t>(bbox->cx) - width / 2;
			std::int32_t top = static_cast<std::int32_t>(bbox->cy) - height / 2;
			
			std::pair<std::size_t, float> bestClassInfo = FindMaxIndexValue(start_class_conf, num_classes);
			result.push_back({left, top, width, height, bestClassInfo.first, bestClassInfo.second});
		}
		*/
		return result;
	}
	cv::Size Yolov5PrePostProcessor::GetCurrentShape() const
	{
		const std::vector<int64_t>& input_shape = m_model_io_info->input.shapes.at(0);
		return { static_cast<int>(input_shape.at(3)), static_cast<int>(input_shape.at(2)) };
	}

	cv::Size Yolov5PrePostProcessor::GetOriginalShape() const
	{
		return m_original_shape;
	}
	std::vector<int64_t> Yolov5PrePostProcessor::GetInputTensorShape() const
	{
		return m_model_io_info->input.shapes.at(0);
	}
}