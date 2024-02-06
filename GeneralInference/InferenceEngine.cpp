#include "InferenceEngine.h"

#include <algorithm>

#include "Yolov5PrePostProcessor.h"
#include "Yolov8PrePostProcessor.h"

#include "Utils.h"

//#define BENCHMARK

#ifdef BENCHMARK
#include <chrono>
#endif // BENCHMARK


namespace GeneralInference
{


	InferenceEngine::InferenceEngine(const std::string& model_path, ModelType model_type)
	{
		
		if (!InitailizeOrt(model_path))
			throw std::runtime_error("Failed to InitailizeOrt!");
		
		if (!ParseModel())
			throw std::runtime_error("Failed to ParseModel!");

		if(!CreatePrePostProcessor(model_type))
			throw std::runtime_error("Failed to InitailizeOrt!");		
	}

	InferenceEngine::~InferenceEngine()
	{

	}

	std::vector<BoundingBox> InferenceEngine::Infer(const cv::Mat& image)
	{
#ifdef BENCHMARK
		auto t1 = std::chrono::high_resolution_clock::now();
#endif
		std::vector<Ort::Value> input_tensor = m_pre_post_processor->Preprocess(image);

#ifdef BENCHMARK
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "Preprocess time:" << 
			std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< "ms\n";
#endif
		std::vector<Ort::Value> output_tensor = Infer(input_tensor);

#ifdef BENCHMARK
		t1 = std::chrono::high_resolution_clock::now();

		std::cout << "Inference time:" <<
			std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count()
			<< "ms\n";
#endif

		return m_pre_post_processor->Postprocess(output_tensor);
	}

	std::vector<BoundingBox> InferenceEngine::Infer(const std::string& image_path)
	{
		cv::Mat image = cv::imread(image_path);
		return Infer(image);
	}

	cv::Mat InferenceEngine::RenderBox(const cv::Mat& image, const std::vector<BoundingBox>& result)
	{
		return RenderBoundingBoxes(image, result, m_model_info->labels);
	}

	std::vector<Ort::Value> InferenceEngine::Infer(const std::vector<Ort::Value>& input_tensor)
	{
		const auto& input_names = m_model_info->input.names_ptr;
		const auto& output_names = m_model_info->output.names_ptr;
		
		return m_ort_session->Run(Ort::RunOptions{ nullptr }, 
			input_names.data(), 
			input_tensor.data(), input_tensor.size(),
			output_names.data(), output_names.size());
	}

	bool InferenceEngine::CreatePrePostProcessor(ModelType model_type)
	{

		switch (model_type)
		{
		case GeneralInference::M_YOLOV5:
		{
			m_pre_post_processor = std::make_shared<Yolov5PrePostProcessor>(m_model_info);
			break;
		}
			
		case GeneralInference::M_YOLOV8:
		{
			m_pre_post_processor = std::make_shared<Yolov8PrePostProcessor>(m_model_info);
			break;
		}	
		default:
			return false;
		}

		return true;
	}

	bool InferenceEngine::IsGPUAvailable()
	{
		std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
		auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
			"CUDAExecutionProvider");

		return cudaAvailable != availableProviders.end();
	}


	bool InferenceEngine::InitailizeOrt(const std::string& model_path)
	{
		m_ort_env_name = model_path;
		m_wmodel_path = S2WS(model_path);
		Ort::SessionOptions ort_session_opt;
		

		m_ort_env = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, m_ort_env_name.c_str());
		
		ort_session_opt.SetIntraOpNumThreads(0);
		ort_session_opt.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
		
		if (IsGPUAvailable()) {
			OrtCUDAProviderOptions cudaOption{};
			ort_session_opt.AppendExecutionProvider_CUDA(cudaOption);
		}
		
		m_ort_session = std::make_shared<Ort::Session>(*m_ort_env, m_wmodel_path.c_str(), ort_session_opt);
		
		return true;
	}
	

	bool InferenceEngine::ParseModel()
	{
		m_model_info = std::make_shared<Model>();
		Ort::AllocatorWithDefaultOptions allocator;
		std::size_t inputCnt = 0, outputCnt = 0;
		
		inputCnt = m_ort_session->GetInputCount();
		outputCnt = m_ort_session->GetOutputCount();
		if (inputCnt != 1 || outputCnt != 1)
			return false;

		
		for (std::size_t idx = 0; idx < inputCnt; ++idx)
		{
			auto name = m_ort_session->GetInputNameAllocated(idx, allocator);
			auto TypeInfo = m_ort_session->GetInputTypeInfo(idx);
			auto TypeAndShape = TypeInfo.GetTensorTypeAndShapeInfo();

			m_model_info->input.Append(name.get(), TypeAndShape.GetShape());
		}

		for (std::size_t idx = 0; idx < outputCnt; ++idx)
		{
			auto name = m_ort_session->GetOutputNameAllocated(idx, allocator);
			auto TypeInfo = m_ort_session->GetOutputTypeInfo(idx);
			auto TypeAndShape = TypeInfo.GetTensorTypeAndShapeInfo();

			m_model_info->output.Append(name.get(), TypeAndShape.GetShape());
		}

		
		m_model_info->labels = ModelParser::ParseLabels(m_ort_session.get());

		return true;
	}
}