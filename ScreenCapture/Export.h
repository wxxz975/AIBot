#pragma once
#include "Capture.h"

#include <opencv2/opencv.hpp>



#define DLL_EXPORT __declspec(dllexport)



extern "C" {
    DLL_EXPORT bool Initialize(int width, int height);


    /// <summary>
    /// 获取正中间的矩形图像
    /// </summary>
    /// <returns>有可能会返回空的cv::Mat, 如果失败还是返回上一次的结果 </returns>
    DLL_EXPORT  bool CaptureNext(cv::Mat& img);
}


static WinDesktopDup capturer;


inline DLL_EXPORT bool Initialize(int width, int height)
{
    return capturer.Initialize(width, height).empty();
}


inline DLL_EXPORT  bool CaptureNext(cv::Mat&img)
{
    if (capturer.CaptureNext()) {
        img = cv::Mat(capturer.Latest.Height, capturer.Latest.Width, CV_8UC4, capturer.Latest.Buf.data());

        return true;
    }

    return false;
}

