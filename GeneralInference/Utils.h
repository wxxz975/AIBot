#pragma once
#include <string>
#include <codecvt>
#include <vector>
#include <opencv2/opencv.hpp>

#include "GeneralStruct.h"

namespace GeneralInference
{
	/// <summary>
	/// Convert std::string to std::wstring
	/// </summary>
	/// <param name="value"></param>
	/// <returns></returns>
	inline std::wstring S2WS(const std::string& value) {
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		return converter.from_bytes(value);
	}

	/// <summary>
	/// Convert std::wstring to std::string
	/// </summary>
	/// <param name="value"></param>
	/// <returns></returns>
	inline std::string WS2S(const std::wstring& value) {
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		return converter.to_bytes(value);
	}


	template<class T>
	inline T Product(const std::vector<T>& values) {
		T result = 1;
		for (const T& value : values)
			result *= value;

		return result;
	}


	cv::Mat Letterbox(const cv::Mat& image, const cv::Size& newShape = cv::Size(640, 640),
		const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scaleFill = false, bool scaleUp = true,
		int stride = 32);


	void Letterbox(const cv::Mat& image, cv::Mat& outImage,
		const cv::Size& newShape,
		const cv::Scalar& color,
		bool auto_,
		bool scaleFill,
		bool scaleUp,
		int stride);


	/// <summary>
	/// Intersection over Union
	/// </summary>
	/// <param name="box1"></param>
	/// <param name="box2"></param>
	/// <returns></returns>
	float IoU(const BoundingBox& box1, const BoundingBox& box2);

	/// <summary>
	/// Non-maximum supression
	/// </summary>
	/// <param name="boxes"></param>
	/// <param name="threshold"></param>
	/// <returns></returns>
	std::vector<BoundingBox> NMS(const std::vector<BoundingBox>& boxes, float threshold);
}