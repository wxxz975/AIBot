#pragma once

#include <string>
#include <opencv2/opencv.hpp>


class ScreenShot
{
public:
	typedef bool(*Initialize)(int width, int height);

	typedef bool (*CaptureNext)(cv::Mat& img);

public:

	ScreenShot();
	~ScreenShot();


	bool Init(int width , int height);


	bool IsValid() const;

	const cv::Mat& Capture();

private:
	std::string dllName = "ScreenCapture.dll";

	Initialize m_initAddr = nullptr;
	CaptureNext m_capAddr = nullptr;

	//HMODULE hDll = nullptr;

	cv::Mat img;
};

