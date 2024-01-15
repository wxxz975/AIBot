#pragma once
#include "InferenceEngine.h"


#define DLL_EXPORT __declspec(dllexport)


//static InferenceEngine engine;


extern "C" {
    /// 这里写导出函数

    DLL_EXPORT bool Initialize(const std::string& engine_path);
    
   DLL_EXPORT void Infer(const cv::Mat& img, std::vector<ResultNode>& results);

    DLL_EXPORT cv::Mat Render(const cv::Mat& img, const std::vector<ResultNode>& results);
}

static InferenceEngine engine;


inline DLL_EXPORT bool Initialize(const std::string& engine_path)
{
    return engine.Initialize(engine_path);
}


inline DLL_EXPORT void Infer(const cv::Mat& img, std::vector<ResultNode>& results)
{
    results = engine.Infer(img);
}


inline DLL_EXPORT cv::Mat Render(const cv::Mat& img, const std::vector<ResultNode>& results)
{
    return engine.Render(img, results);
}

