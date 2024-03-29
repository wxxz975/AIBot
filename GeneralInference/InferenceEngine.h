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


		/// <summary>
		/// Draw the detection results onto the image and return the drawn image
		/// </summary>
		/// <param name="image"></param>
		/// <param name="result"></param>
		/// <returns></returns>
		cv::Mat RenderBox(const cv::Mat& image, const std::vector<BoundingBox>& result);

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


		std::shared_ptr<Ort::Session>  m_ort_session;
		std::shared_ptr<Ort::Env> m_ort_env;

		std::string m_ort_env_name;
		std::wstring m_wmodel_path;

		float confidence_threshold = 0.5;
		float iou_threshold = 0.5;
	};

	

}