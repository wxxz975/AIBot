#include "Export.h"




/*
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        
        break;
    }

    return TRUE;
}*/
#include <chrono>
int main(int argc, char* argv[])
{
   
    capturer.Initialize(640, 640);

    /*
        就是这里太慢了
    */
    cv::Mat image;
    std::size_t total_capture = 10000000;
    std::size_t success_cnt = 0;
    
    auto start = std::chrono::steady_clock::now();
    for (std::size_t idx = 0; idx < total_capture; ++idx) {
        if (capturer.CaptureNext())
            success_cnt++;
    }

    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "total times:" << total_capture << " success:" << success_cnt 
        << " total time:" << ms
        << "ms " << " fps:" << 1000 /(ms / success_cnt) 
        << "\n";


}