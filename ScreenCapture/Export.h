#pragma once
#include "Capture.h"

#include <opencv2/opencv.hpp>



#define DLL_EXPORT __declspec(dllexport)



extern "C" {
    DLL_EXPORT bool Initialize(int width, int height);


    /// <summary>
    /// ��ȡ���м�ľ���ͼ��
    /// </summary>
    /// <returns>�п��ܻ᷵�ؿյ�cv::Mat, ���ʧ�ܻ��Ƿ�����һ�εĽ�� </returns>
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

