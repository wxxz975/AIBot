#pragma once

#include "PrePostProcessor.h"
#include "ModelParser.h"

namespace GeneralInference
{
	enum ModelType
	{
		M_YOLOV5 = 0,
		M_YOLOV8,
		M_MAX_TYPE
	};

	class InferenceEngine
	{
	public:
		/// <summary>
		/// 
		/// </summary>
		/// <param name="model_path"></param>
		/// <param name="model_type"></param>
		InferenceEngine(const std::string& model_path, ModelType model_type = M_YOLOV5);
		~InferenceEngine();

		/// <summary>
		/// 
		/// </summary>
		/// <param name="image"></param>
		/// <returns></returns>
		std::vector<BoundingBox> Infer(const cv::Mat& image);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="image_path"></param>
		/// <returns></returns>
		std::vector<BoundingBox> Infer(const std::string& image_path);

	private:

		std::vector<Ort::Value> Infer(const std::vector<Ort::Value>& input_tensor);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="model_type"></param>
		/// <returns></returns>
		bool CreatePrePostProcessor(ModelType model_type);

		/// <summary>
		/// check the gpu available or not
		/// </summary>
		/// <returns></returns>
		bool IsGPUAvailable();


		/// <summary>
		/// 
		/// </summary>
		void WarmUpModel();


		/// <summary>
		/// 
		/// </summary>
		/// <param name="model_path"></param>
		/// <returns></returns>
		bool InitailizeOrt(const std::string& model_path);

		
		bool ParseModel();


	private:

		std::shared_ptr<PrePostProcessor> m_pre_post_processor;
		std::shared_ptr<Model> m_model_info;


		Ort::Session m_ort_session{nullptr};
		Ort::Env m_ort_env;
		std::string m_ort_env_name;

		float confidence_threshold = 0.5;
		float iou_threshold = 0.5;
	};

	

}