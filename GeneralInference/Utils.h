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


	/*
	void Letterbox(const cv::Mat& image, cv::Mat& outImage,
		const cv::Size& newShape,
		const cv::Scalar& color = cv::Scalar(114,114,114),
		bool auto_,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32);
		*/
	/// <summary>
	/// Restore the inferred coordinates to the coordinates of the original image
	/// </summary>
	/// <param name="currentShape"></param>
	/// <param name="originalShape"></param>
	/// <param name="bbox"></param>
	void RestoreOriginalCoordinates(const cv::Size& currentShape, const cv::Size& originalShape, BoundingBox& bbox);

	void RestoreOriginalCoordsInBatch(const cv::Size& currentShape, const cv::Size& originalShape, std::vector<BoundingBox>& boxes);

	/// <summary>
	/// Intersection over Union
	/// </summary>
	/// <param name="box1">left box</param>
	/// <param name="box2">right box</param>
	/// <returns></returns>
	float IoU(const BoundingBox& box1, const BoundingBox& box2);

	/// <summary>
	/// Non-maximum supression, Confidence thresholds and coordinate restoration are not handled here.
	/// </summary>
	/// <param name="boxes"></param>
	/// <param name="threshold"></param>
	/// <returns></returns>
	std::vector<BoundingBox> NMS(const std::vector<BoundingBox>& boxes, float iou_threshold);


	/// <summary>
	/// Get the category with the best confidence, return the category id and confidence
	/// </summary>
	/// <param name="iter"></param>
	/// <param name="num_class"></param>
	/// <returns></returns>
	std::pair<std::size_t, float> FindMaxIndexValue(std::vector<float>::iterator& iter, std::size_t num_class);
	

	/// <summary>
	/// Convert gray, bgr, bgra type image to rgb image
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	cv::Mat Convert2RGB(const cv::Mat& image);


	/// <summary>
	/// Convert image from HWC format to CHW format 
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	void HWC2CHW(const cv::Mat& image, std::vector<float>& blob);

	void HWC2CHW(const cv::Mat& image, float* blob);


	/// <summary>
	/// Draw the target box into the image
	/// </summary>
	/// <param name="image"></param>
	/// <param name="boxes"></param>
	/// <param name="labels"></param>
	/// <returns></returns>
	cv::Mat RenderBoundingBoxes(const cv::Mat& image, const std::vector<BoundingBox>& boxes,
		const std::vector<std::string>& labels = std::vector<std::string>());

}