#include "InferenceEngine.h"
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "ISession.h"
#include "Prepostprocessor.h"


/*
	ע������������onnxruntime��onnxruntime_providers_cuda.dll���dll�� ����onnxruntime_providers_cuda.dll������
*/

class InferenceEngine: public ISession
{
public:
	InferenceEngine();
	~InferenceEngine();

	/// <summary>
	/// ��ʼ�����engine�� 
	/// </summary>
	/// <param name="model_path"></param>
	/// <returns></returns>
	bool Initialize(const std::string& model_path) override;

	/// <summary>
	/// ��������
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
	/// ���ڵ���ʱ���ӻ����������Ч�����������Ŀ���Ƶ�ͼ���ϣ����ػ��ƺ��ͼ��
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
