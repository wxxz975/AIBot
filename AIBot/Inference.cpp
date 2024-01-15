#include "Inference.h"
#include <fstream>
#include <Windows.h>

#include "Logger.h"

Inference::Inference()
{
}

Inference::~Inference()
{

}

bool Inference::Init(const std::string& engine_path)
{
    std::ifstream file(engine_path);
    if (!file.good()){
        // logging
        return false;
    }

    HMODULE hDll = LoadLibraryA(dllName.c_str());
    if (hDll == INVALID_HANDLE_VALUE)
    {
        err("Failed to load DLL:%s, %d", dllName.c_str(), GetLastError());
        return false;
    }

    m_initAddr = reinterpret_cast<Initialize>(GetProcAddress(hDll, "Initialize"));
    m_inferAddr = reinterpret_cast<Infer>(GetProcAddress(hDll, "Infer"));
    m_renderAddr = reinterpret_cast<Render>(GetProcAddress(hDll, "Render"));
    

    log("m_initAddr:%llx, m_inferAddr:%llx, m_renderAddr:%llx\n", m_initAddr, m_inferAddr, m_renderAddr);

    return IsValid() && m_initAddr(engine_path);
}

bool Inference::IsValid() const
{
    return m_initAddr && m_inferAddr && m_renderAddr;
}

const std::vector<ResultNode>& Inference::doInfer(const cv::Mat& img)
{
    results.clear();
    m_inferAddr(img, results);
    
    return results;
}

cv::Mat Inference::doRender(const cv::Mat& img, const std::vector<ResultNode>& results)
{
    return m_renderAddr(img, results);
}
