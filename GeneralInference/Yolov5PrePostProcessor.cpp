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

		// Convert image to rgb format
		rgb_image = Convert2RGB(image);

		// Convert the image to an image of a specified size using letterbox
		letterbox_image = Letterbox(rgb_image, GetCurrentShape());
		
		// Convert uint8_t type to float32 type and normalize to between 0-1
		letterbox_image.convertTo(float_image, CV_32FC3, 1 / 255.0);

		// Convert image color channel from hwc to chw
		HWC2CHW(float_image, m_blob);
		

		result.push_back(
			Ort::Value::CreateTensor<float>(m_ort_memory_info,
				m_blob.data(), m_blob.size(),
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
		const std::vector<Ort::Value>& output_tensor, float conf_threshlod)
	{
		std::vector<BoundingBox> result;
		// output only one Dimensions
		const float* raw_ptr = output_tensor.at(0).GetTensorData<float>();
		std::size_t count = output_tensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<int64_t> output_shape = output_tensor.at(0).GetTensorTypeAndShapeInfo().GetShape();
		std::vector<float> output(raw_ptr, raw_ptr + count);
		
		// These all depend on the model, [1,25200,19]
		std::size_t size_bbox = output_shape.at(2);
		std::size_t num_classes = size_bbox - YOLOV5_OUTBOX_ELEMENT_COUNT;

		for (auto iter = output.begin(); iter != output.begin() + count; iter += size_bbox)
		{
			const YoloRawResult* bbox = reinterpret_cast<const YoloRawResult*>(&(*iter));
			float boxConf = bbox->box_conf; // box confidence
			auto start_class_conf = iter + YOLOV5_OUTBOX_ELEMENT_COUNT;

			if (boxConf < conf_threshlod)
				continue;

			std::int32_t width = (std::int32_t)(bbox->w);
			std::int32_t height = (std::int32_t)(bbox->h);
			std::int32_t left = (std::int32_t)(bbox->cx) - width / 2;
			std::int32_t top = (std::int32_t)(bbox->cy) - height / 2;
			
			std::pair<std::size_t, float> bestClassInfo = FindMaxIndexValue(start_class_conf, num_classes);
			result.push_back({left, top, width, height, bestClassInfo.first, bestClassInfo.second});
		}

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