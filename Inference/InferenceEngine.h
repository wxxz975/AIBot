#include "InferenceEngine.h"
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "ISession.h"
#include "Prepostprocessor.h"


/*
	注：本程序依赖onnxruntime、onnxruntime_providers_cuda.dll相关dll， 但是onnxruntime_providers_cuda.dll又依赖
*/

class InferenceEngine: public ISession
{
public:
	InferenceEngine();
	~InferenceEngine();

	/// <summary>
	/// 初始化这个engine， 
	/// </summary>
	/// <param name="model_path"></param>
	/// <returns></returns>
	bool Initialize(const std::string& model_path) override;

	/// <summary>
	/// 推理的入口
	/// </summary>
	/// <param name="img"></param>
	/// <returns></returns>
	std::vector<ResultNode> Infer(const cv::Mat& img) override;
private:

	bool CreateSession(const std::string& modelPath);

	bool ParseModel();

	OrtCUDAProviderOptions CreateCudaOptions();

	bool IsGPUAvailable();
public:
	/// <summary>
	/// 用于调试时可视化这个推理后的效果，将推理后的框绘制到图像上，返回绘制后的图像
	/// </summary>
	/// <param name="img"></param>
	/// <param name="results"></param>
	/// <returns></returns>
	cv::Mat Render(const cv::Mat& img, const std::vector<ResultNode>& results);

private:
	inline bool WarmUpModel() { return true;/* nothing todo */ }
	

private:

	Prepostprocessor* processor_ = nullptr;
	bool isGpuAval = false;
};
