#pragma once

#include <string>
#include <opencv2/opencv.hpp>

typedef struct _DetectionResultNode
{
	float x, y, w, h;
	int classIdx;
	float confidence;
}ResultNode, * pResultNode;

class Inference
{
private:
	typedef bool(*Initialize)(const std::string& engine_path);
	
	//void Infer(const cv::Mat& img, std::vector<ResultNode>& results);
	typedef void(*Infer)(const cv::Mat& img, std::vector<ResultNode>& results);

	typedef cv::Mat (*Render)(const cv::Mat& img, const std::vector<ResultNode>& results);

public:
	Inference();
	~Inference();

	bool Init(const std::string& engine_path);

	bool IsValid() const;

	const std::vector<ResultNode>& doInfer(const cv::Mat& img);

	cv::Mat doRender(const cv::Mat& img, const std::vector<ResultNode>& results);

private:
	
	std::string dllName = "Inference.dll";


	Initialize m_initAddr = nullptr;
	Infer m_inferAddr = nullptr;
	Render m_renderAddr = nullptr;

	std::vector<ResultNode> results;

};

